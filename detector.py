# hotspot_detector.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class CongestionHotspotDetector:
    """检测和可视化拥塞热点的类"""
    
    def __init__(self, threshold=0.7, min_area=50, highlight_color=(255, 0, 0)):
        """
        初始化拥塞热点检测器
        
        参数:
            threshold: 拥塞值阈值，高于此值的区域被视为热点
            min_area: 热点的最小面积（像素数），过滤掉太小的区域
            highlight_color: 热点高亮颜色，默认为红色 (BGR格式)
        """
        self.threshold = threshold
        self.min_area = min_area
        self.highlight_color = highlight_color
    
    def detect_from_file(self, png_file_path, output_dir=None):
        """
        从PNG文件检测热点并保存结果
        
        参数:
            png_file_path: 拥塞图像的文件路径
            output_dir: 输出目录，如果为None则使用输入文件所在目录
        
        返回:
            热点标注后的图像路径
        """
        # 读取图像
        image = cv2.imread(png_file_path)
        if image is None:
            raise ValueError(f"无法读取图像: {png_file_path}")
        
        # 检测热点
        hotspots, annotated_image = self.detect_hotspots(image)
        
        # 保存结果
        if output_dir is None:
            output_dir = os.path.dirname(png_file_path)
        
        base_name = os.path.basename(png_file_path)
        filename, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{filename}_hotspots{ext}")
        
        cv2.imwrite(output_path, annotated_image)
        
        # 生成热点报告
        report_path = os.path.join(output_dir, f"{filename}_hotspot_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"拥塞热点分析报告 - {base_name}\n")
            f.write(f"检测阈值: {self.threshold}\n")
            f.write(f"检测到的热点数量: {len(hotspots)}\n\n")
            
            for i, (contour, area, severity) in enumerate(hotspots):
                x, y, w, h = cv2.boundingRect(contour)
                f.write(f"热点 #{i+1}:\n")
                f.write(f"  位置: X={x}, Y={y}, 宽={w}, 高={h}\n")
                f.write(f"  面积: {area} 像素\n")
                f.write(f"  严重度: {severity:.2f}\n\n")
        
        return output_path
    
    def detect_hotspots(self, image):
        """
        检测给定图像中的拥塞热点
        
        参数:
            image: 拥塞图像 (OpenCV格式)
        
        返回:
            hotspots: 热点列表，每个热点为 (轮廓, 面积, 严重度) 的元组
            annotated_image: 带有热点标注的图像
        """
        # 创建图像副本用于标注
        annotated_image = image.copy()
        
        # 转换为HSV颜色空间，便于分析颜色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 提取红色和橙色区域(对应高拥塞区域)
        # 红色区域 (两个范围，因为红色在HSV中横跨了色环的两端)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # 橙色区域
        lower_orange = np.array([11, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # 合并所有高拥塞掩码
        high_congestion_mask = cv2.bitwise_or(cv2.bitwise_or(mask_red1, mask_red2), mask_orange)
        
        # 使用形态学操作连接接近的区域
        kernel = np.ones((3, 3), np.uint8)
        high_congestion_mask = cv2.morphologyEx(high_congestion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 找到轮廓
        contours, _ = cv2.findContours(high_congestion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小区域并计算热点严重度
        hotspots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            # 创建掩码来提取该轮廓区域
            mask = np.zeros_like(high_congestion_mask)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # 提取对应区域的HSV值来估计严重度
            region_hsv = hsv[mask == 255]
            if len(region_hsv) > 0:
                # 使用色调来估计严重度 (红色最高，橙色次之)
                hues = region_hsv[:, 0]
                # 将色调转换为严重度分数 (0-1范围)
                severity_score = 1.0 - (np.mean(hues) / 30.0 if np.mean(hues) < 30 else 0.5)
                severity_score = min(1.0, max(0.5, severity_score))  # 限制在0.5-1.0范围内
            else:
                severity_score = 0.5
            
            hotspots.append((contour, area, severity_score))
        
        # 按严重度排序
        hotspots.sort(key=lambda x: x[2], reverse=True)
        
        # 在图像上标记热点
        for i, (contour, area, severity) in enumerate(hotspots):
            # 绘制轮廓
            cv2.drawContours(annotated_image, [contour], 0, self.highlight_color, 2)
            
            # 添加标签
            x, y, w, h = cv2.boundingRect(contour)
            label = f"#{i+1} ({severity:.2f})"
            cv2.putText(annotated_image, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.highlight_color, 2)
            
            # 对于非常严重的热点(>=0.8)，添加额外警告标志
            if severity >= 0.8:
                center = (x + w // 2, y + h // 2)
                radius = min(w, h) // 4
                cv2.circle(annotated_image, center, radius, (0, 0, 255), -1)
                cv2.circle(annotated_image, center, radius, (255, 255, 255), 2)
                
                warning_size = cv2.getTextSize("!", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                warning_x = center[0] - warning_size[0] // 2
                warning_y = center[1] + warning_size[1] // 2
                cv2.putText(annotated_image, "!", (warning_x, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return hotspots, annotated_image
    
    def detect_from_data(self, congestion_matrix, output_path, title=None):
        """
        直接从拥塞矩阵数据检测热点
        
        参数:
            congestion_matrix: 拥塞数据矩阵 (numpy array)
            output_path: 输出图像路径
            title: 图像标题，默认为None
            
        返回:
            热点标注后的图像路径
        """
        # 创建一个基于数据的热力图
        plt.figure(figsize=(12, 10))
        
        # 绘制热力图
        plt.imshow(congestion_matrix, cmap='jet', interpolation='none', origin='lower')
        
        # 计算热点区域 (>= threshold的区域)
        hotspot_mask = congestion_matrix >= self.threshold
        
        # 标记热点区域
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(hotspot_mask)
        
        # 统计每个热点的属性
        hotspots = []
        for i in range(1, num_features + 1):
            # 获取该热点区域掩码
            region_mask = labeled_mask == i
            
            # 计算面积
            area = np.sum(region_mask)
            if area < self.min_area:
                continue
                
            # 计算严重度
            severity = np.mean(congestion_matrix[region_mask])
            
            # 获取边界坐标
            y_indices, x_indices = np.where(region_mask)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            
            hotspots.append({
                'area': area,
                'severity': severity,
                'bounds': (min_x, min_y, max_x - min_x, max_y - min_y),
                'centroid': (np.mean(x_indices), np.mean(y_indices))
            })
        
        # 在图上标注热点
        for i, hotspot in enumerate(hotspots):
            if hotspot['area'] < self.min_area:
                continue
                
            x, y, w, h = hotspot['bounds']
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, 
                                             edgecolor='red', linewidth=2))
            
            # 添加标签
            cx, cy = hotspot['centroid']
            plt.text(cx, cy, f"#{i+1}\n{hotspot['severity']:.2f}", 
                    color='white', fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round", fc="red", alpha=0.7))
        
        if title:
            plt.title(title)
        plt.colorbar(label='Congestion Value')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        # 生成热点报告
        report_path = os.path.splitext(output_path)[0] + "_hotspot_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"拥塞热点分析报告\n")
            f.write(f"检测阈值: {self.threshold}\n")
            f.write(f"检测到的热点数量: {len(hotspots)}\n\n")
            
            for i, hotspot in enumerate(hotspots):
                x, y, w, h = hotspot['bounds']
                f.write(f"热点 #{i+1}:\n")
                f.write(f"  位置: X={x}, Y={y}, 宽={w}, 高={h}\n")
                f.write(f"  面积: {hotspot['area']} 像素\n")
                f.write(f"  严重度: {hotspot['severity']:.2f}\n\n")
        
        return output_path

    def detect_hotspots_from_raw_data(self, raw_data, output_path=None, title=None):
        """
        直接从原始拥塞数据检测热点
        
        参数:
            raw_data: 原始拥塞数据矩阵，形状为(rows, cols)
            output_path: 输出图像的路径，如果为None则不保存图像
            title: 图像标题
            
        返回:
            hotspots: 热点列表，每个热点包含位置、面积和严重度信息
        """
        # 保存数据维度
        self.data_rows, self.data_cols = raw_data.shape
        print(f"原始数据维度: {self.data_rows} 行 x {self.data_cols} 列")
        
        # 直接使用阈值检测热点
        hotspot_mask = raw_data >= self.threshold
        
        # 标记连通区域
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(hotspot_mask)
        print(f"初步检测到 {num_features} 个连通区域")
        
        # 处理每个连通区域
        hotspots = []
        for i in range(1, num_features + 1):
            # 获取当前区域的掩码
            region_mask = labeled_mask == i
            
            # 计算面积
            area = np.sum(region_mask)
            
            # 过滤掉太小的区域
            if area < self.min_area:
                continue
                
            # 计算严重度 (该区域内的平均拥塞值)
            severity = np.mean(raw_data[region_mask])
            
            # 获取边界框 (行列格式)
            rows, cols = np.where(region_mask)
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            
            # 计算质心
            centroid_row = np.mean(rows)
            centroid_col = np.mean(cols)
            
            # 添加到热点列表
            hotspot = {
                'bounds': (min_row, min_col, height, width),  # (row, col, height, width)
                'centroid': (centroid_row, centroid_col),     # (row, col)
                'area': area,
                'severity': severity,
                'region_mask': region_mask  # 保存掩码以便后续分析
            }
            hotspots.append(hotspot)
        
        print(f"检测到 {len(hotspots)} 个有效热点 (面积 >= {self.min_area})")
        
        # 按严重度排序
        hotspots.sort(key=lambda x: x['severity'], reverse=True)
        
        # 可视化热点并生成报告
        if output_path:
            # 可视化热点
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 10))
            
            # 显示原始数据
            plt.imshow(raw_data, cmap='jet', interpolation='none')
            
            # 标记热点
            for i, hotspot in enumerate(hotspots):
                row, col, h, w = hotspot['bounds']
                
                # 在热点周围画矩形 (matplotlib中x对应col，y对应row)
                rect = plt.Rectangle((col, row), w, h, fill=False, 
                                   edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # 添加标签
                c_row, c_col = hotspot['centroid']
                label = f"#{i+1}\n{hotspot['severity']:.2f}"
                plt.text(c_col, c_row, label, color='white', fontsize=9, 
                        ha='center', va='center', 
                        bbox=dict(boxstyle="round", fc="red", alpha=0.7))
            
            if title:
                plt.title(title)
            plt.colorbar(label='Congestion Value')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            # 生成热点报告
            report_path = os.path.splitext(output_path)[0] + "_hotspot_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"拥塞热点分析报告\n")
                if title:
                    f.write(f"数据来源: {title}\n")
                f.write(f"检测阈值: {self.threshold}\n")
                f.write(f"原始数据维度: {self.data_rows} 行 x {self.data_cols} 列\n")
                f.write(f"检测到的热点数量: {len(hotspots)}\n\n")
                
                for i, hotspot in enumerate(hotspots):
                    row, col, height, width = hotspot['bounds']
                    
                    f.write(f"热点 #{i+1}:\n")
                    f.write(f"  原始数据位置: 行={row}, 列={col}, 高={height}, 宽={width}\n")
                    f.write(f"  图像位置: X={col}, Y={row}, 高={height}, 宽={width}\n")
                    f.write(f"  面积: {hotspot['area']} 单元格\n")
                    f.write(f"  严重度: {hotspot['severity']:.2f}\n\n")
        
        return hotspots

    def generate_visualization_with_hotspots(self, raw_data, output_path, title=None):
        """
        生成带有热点标记的可视化，并同时提供原始数据和图像坐标的报告
        
        参数:
            raw_data: 原始拥塞数据矩阵
            output_path: 输出图像路径
            title: 图像标题
        """
        # 首先检测热点
        hotspots = self.detect_hotspots_from_raw_data(raw_data, None, title)
        
        # 创建高质量可视化
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        # 创建自定义色图，使高拥塞区域更加明显
        cmap = LinearSegmentedColormap.from_list(
            'congestion_cmap', 
            [(0, 'blue'), (0.5, 'green'), (0.7, 'yellow'), (0.8, 'orange'), (1, 'red')]
        )
        
        plt.figure(figsize=(16, 12))
        plt.imshow(raw_data, cmap=cmap, interpolation='bilinear')
        
        # 添加热点标记
        for i, hotspot in enumerate(hotspots):
            row, col, h, w = hotspot['bounds']
            
            # 添加矩形边框
            color = 'red' if hotspot['severity'] >= 0.8 else 'white'
            linewidth = 3 if hotspot['severity'] >= 0.8 else 2
            rect = plt.Rectangle((col, row), w, h, fill=False, 
                               edgecolor=color, linewidth=linewidth)
            plt.gca().add_patch(rect)
            
            # 添加标签
            c_row, c_col = hotspot['centroid']
            label = f"#{i+1} ({hotspot['severity']:.2f})"
            plt.text(c_col, c_row, label, color='white', fontsize=10, 
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round", fc="red", alpha=0.7))
        
        # 添加标题和颜色条
        if title:
            plt.title(title, fontsize=14)
        cbar = plt.colorbar(label='Congestion Value')
        cbar.ax.tick_params(labelsize=12)
        
        # 添加坐标轴标签
        plt.xlabel('Column (X)', fontsize=12)
        plt.ylabel('Row (Y)', fontsize=12)
        
        # 保存高分辨率图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成详细报告，包含原始数据坐标和图像坐标
        report_path = os.path.splitext(output_path)[0] + "_hotspot_report.txt"
        
        # 获取图像尺寸 (为了计算图像坐标)
        # 假设图像按照标准比例缩放
        img_width = 1000  # 假设的图像宽度
        img_height = int(img_width * (self.data_rows / self.data_cols))  # 按比例计算高度
        
        with open(report_path, 'w') as f:
            f.write(f"拥塞热点分析报告\n")
            if title:
                f.write(f"数据来源: {title}\n")
            f.write(f"检测阈值: {self.threshold}\n")
            f.write(f"原始数据维度: {self.data_rows} 行 x {self.data_cols} 列\n")
            f.write(f"检测到的热点数量: {len(hotspots)}\n\n")
            
            for i, hotspot in enumerate(hotspots):
                row, col, height, width = hotspot['bounds']
                
                # 计算对应的图像坐标 (按比例缩放)
                img_x = int(col * (img_width / self.data_cols))
                img_y = int(row * (img_height / self.data_rows))
                img_w = int(width * (img_width / self.data_cols))
                img_h = int(height * (img_height / self.data_rows))
                
                f.write(f"热点 #{i+1}:\n")
                f.write(f"  原始数据位置: 行={row}, 列={col}, 高={height}, 宽={width}\n")
                f.write(f"  图像位置: X={img_x}, Y={img_y}, 宽={img_w}, 高={img_h}\n")
                f.write(f"  面积: {hotspot['area']} 单元格\n")
                f.write(f"  严重度: {hotspot['severity']:.2f}\n\n")
        
        return hotspots


def batch_process_congestion_images(report_dir, output_dir=None, threshold=0.7, min_area=50):
    """
    批量处理报告目录中的所有拥塞图像
    
    参数:
        report_dir: 报告目录路径
        output_dir: 输出目录路径，默认与报告目录相同
        threshold: 拥塞阈值
        min_area: 最小热点面积
    """
    if output_dir is None:
        output_dir = report_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建热点检测器
    detector = CongestionHotspotDetector(threshold=threshold, min_area=min_area)
    
    # 查找所有拥塞图像
    congestion_patterns = [
        "*congestion*.png"
    ]
    
    import glob
    all_files = []
    for pattern in congestion_patterns:
        all_files.extend(glob.glob(os.path.join(report_dir, pattern)))
    
    # 处理每个文件
    results = []
    for file_path in all_files:
        try:
            output_path = detector.detect_from_file(file_path, output_dir)
            results.append((file_path, output_path, "成功"))
            print(f"处理成功: {file_path}")
        except Exception as e:
            results.append((file_path, None, f"错误: {str(e)}"))
            print(f"处理失败: {file_path} - {str(e)}")
    
    # 生成批处理报告
    report_path = os.path.join(output_dir, "batch_processing_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"拥塞热点批处理报告\n")
        f.write(f"处理的文件总数: {len(results)}\n")
        f.write(f"成功数量: {sum(1 for _, _, status in results if status == '成功')}\n")
        f.write(f"失败数量: {sum(1 for _, _, status in results if status != '成功')}\n\n")
        
        f.write("详细处理结果:\n")
        for input_path, output_path, status in results:
            f.write(f"输入文件: {os.path.basename(input_path)}\n")
            f.write(f"状态: {status}\n")
            if output_path:
                f.write(f"输出文件: {os.path.basename(output_path)}\n")
            f.write("\n")
    
    return results


def integrate_with_visualizers(congestion_matrix, header, output_path, threshold=0.7, min_area=50):
    """
    与可视化模块集成的函数，可在生成拥塞图的同时检测热点
    
    参数:
        congestion_matrix: 拥塞数据矩阵
        header: 头信息
        output_path: 输出路径
        threshold: 拥塞阈值
        min_area: 最小热点面积
    """
    # 获取基本文件名
    base_output_path = os.path.splitext(output_path)[0]
    hotspot_output_path = f"{base_output_path}_hotspots.png"
    
    # 创建热点检测器
    detector = CongestionHotspotDetector(threshold=threshold, min_area=min_area)
    
    # 使用矩阵数据直接进行热点检测
    title = header.get('source_file', 'Congestion Hotspot Analysis')
    detector.detect_from_data(congestion_matrix, hotspot_output_path, title)
    
    return hotspot_output_path


if __name__ == "__main__":
    # 使用示例
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python hotspot_detector.py <拥塞图像路径>")
        print("  python hotspot_detector.py --batch <报告目录路径>")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("请提供报告目录路径")
            sys.exit(1)
        
        report_dir = sys.argv[2]
        print(f"批量处理目录: {report_dir}")
        
        # 可选参数
        threshold = 0.7
        min_area = 50
        
        if len(sys.argv) > 3:
            threshold = float(sys.argv[3])
        if len(sys.argv) > 4:
            min_area = int(sys.argv[4])
        
        results = batch_process_congestion_images(report_dir, 
                                                threshold=threshold, 
                                                min_area=min_area)
        print(f"批处理完成，共处理 {len(results)} 个文件")
        
    else:
        image_path = sys.argv[1]
        print(f"处理图像: {image_path}")
        
        # 可选参数
        threshold = 0.7
        min_area = 50
        
        if len(sys.argv) > 2:
            threshold = float(sys.argv[2])
        if len(sys.argv) > 3:
            min_area = int(sys.argv[3])
        
        detector = CongestionHotspotDetector(threshold=threshold, min_area=min_area)
        output_path = detector.detect_from_file(image_path)
        
        print(f"处理完成，结果保存至: {output_path}")