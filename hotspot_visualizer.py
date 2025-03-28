#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hotspot_visualizer.py - 集成到现有可视化流程的热点检测与可视化

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import copy

class HotspotVisualizer:
    """用于检测和可视化拥塞热点的类，可集成到现有流程"""
    
    def __init__(self, threshold=0.7, min_area=50):
        """
        初始化热点可视化器
        
        参数:
            threshold: 拥塞阈值，高于此值的区域被视为热点
            min_area: 最小热点面积（单元格数量）
        """
        self.threshold = threshold
        self.min_area = min_area
    
    def detect_hotspots(self, data_matrix):
        """
        检测热点区域
        
        参数:
            data_matrix: 拥塞数据矩阵
            
        返回:
            hotspots: 热点列表
        """
        # 保存数据维度
        data_rows, data_cols = data_matrix.shape
        
        # 应用阈值
        hotspot_mask = data_matrix >= self.threshold
        
        # 标记连通区域
        labeled_mask, num_features = ndimage.label(hotspot_mask)
        
        # 分析每个连通区域
        hotspots = []
        for i in range(1, num_features + 1):
            # 获取区域掩码
            region_mask = labeled_mask == i
            
            # 计算面积
            area = np.sum(region_mask)
            
            # 过滤太小的区域
            if area < self.min_area:
                continue
                
            # 计算严重度（区域内的平均拥塞值）
            severity = np.mean(data_matrix[region_mask])
            
            # 获取边界框（行列格式）
            rows, cols = np.where(region_mask)
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            
            # 计算质心
            centroid_row = np.mean(rows)
            centroid_col = np.mean(cols)
            
            # 创建热点信息
            hotspot = {
                'bounds': (min_row, min_col, height, width),  # (row, col, height, width)
                'centroid': (centroid_row, centroid_col),     # (row, col)
                'area': area,
                'severity': severity,
                'mask': region_mask,  # 保存掩码以便后续使用
            }
            hotspots.append(hotspot)
        
        # 按严重度排序（从高到低）
        hotspots.sort(key=lambda x: x['severity'], reverse=True)
        
        return hotspots
    
    def create_hotspot_visualization(self, data_matrix, original_output_path, hotspots=None):
        """
        创建热点可视化
        
        参数:
            data_matrix: 拥塞数据矩阵
            original_output_path: 原始可视化的输出路径
            hotspots: 热点列表（如果为None则自动检测）
            
        返回:
            hotspot_output_path: 热点可视化的输出路径
        """
        # 检测热点（如果未提供）
        if hotspots is None:
            hotspots = self.detect_hotspots(data_matrix)
        
        # 创建输出路径
        base_path = os.path.splitext(original_output_path)[0]
        hotspot_output_path = f"{base_path}_hotspots.png"
        
        # 创建热点可视化
        plt.figure(figsize=(12, 10))
        
        # 显示原始数据
        plt.imshow(data_matrix, cmap='jet', interpolation='bilinear')
        
        # 标记热点
        for i, hotspot in enumerate(hotspots):
            row, col, height, width = hotspot['bounds']
            
            # 绘制矩形边框
            severity = hotspot['severity']
            color = 'red' if severity >= 0.8 else ('orange' if severity >= 0.7 else 'white')
            linewidth = 3 if severity >= 0.8 else 2
            
            rect = plt.Rectangle((col, row), width, height, 
                               fill=False, edgecolor=color, linewidth=linewidth)
            plt.gca().add_patch(rect)
            
            # 添加标签
            c_row, c_col = hotspot['centroid']
            label = f"#{i+1}\n{severity:.2f}"
            
            # 为标签添加背景色
            bbox_props = dict(
                boxstyle="round,pad=0.3", 
                fc='red' if severity >= 0.8 else 'orange',
                alpha=0.7
            )
            plt.text(c_col, c_row, label, color='white', 
                    ha='center', va='center', fontsize=9,
                    bbox=bbox_props)
        
        # 添加标题
        plt.title(f"拥塞热点分析 (阈值: {self.threshold}, 热点数: {len(hotspots)})")
        
        # 添加颜色条
        cbar = plt.colorbar(label='Congestion Value')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(hotspot_output_path, dpi=150)
        plt.close()
        
        # 生成热点报告
        self._generate_report(hotspots, data_matrix.shape, hotspot_output_path)
        
        return hotspot_output_path
    
    def _generate_report(self, hotspots, data_shape, output_path):
        """生成热点报告"""
        data_rows, data_cols = data_shape
        report_path = os.path.splitext(output_path)[0] + "_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"拥塞热点分析报告\n")
            f.write(f"检测阈值: {self.threshold}\n")
            f.write(f"数据维度: {data_rows} 行 x {data_cols} 列\n")
            f.write(f"检测到的热点数量: {len(hotspots)}\n\n")
            
            for i, hotspot in enumerate(hotspots):
                row, col, height, width = hotspot['bounds']
                
                f.write(f"热点 #{i+1}:\n")
                f.write(f"  位置: 行={row}, 列={col}, 高={height}, 宽={width}\n")
                f.write(f"  面积: {hotspot['area']} 单元格\n")
                f.write(f"  严重度: {hotspot['severity']:.2f}\n\n")


def visualize_with_hotspots(data_matrix, output_path, header=None, 
                           threshold=0.7, min_area=50):
    """
    创建拥塞图并同时生成热点分析版本
    
    参数:
        data_matrix: 拥塞数据矩阵
        output_path: 输出路径
        header: 数据头信息
        threshold: 热点检测阈值
        min_area: 最小热点面积
        
    返回:
        tuple: (原始输出路径, 热点分析输出路径)
    """
    # 导入visualizers模块以复用现有功能
    from visualizers import visualize_heatmap
    
    # 创建常规的拥塞图
    original_path = visualize_heatmap(data_matrix, output_path, header)
    
    # 创建热点分析版本
    hotspot_visualizer = HotspotVisualizer(threshold=threshold, min_area=min_area)
    hotspots = hotspot_visualizer.detect_hotspots(data_matrix)
    hotspot_path = hotspot_visualizer.create_hotspot_visualization(
        data_matrix, original_path, hotspots)
    
    return original_path, hotspot_path


# 集成到现有可视化流程
def integrate_with_visualizers():
    """
    修改visualizers.py中的函数，集成热点检测功能
    """
    import visualizers
    original_visualize_congestion = visualizers.visualize_congestion
    
    def enhanced_visualize_congestion(congestion_data, output_dir, label_postfix="", 
                                     threshold=0.7, min_area=50):
        """增强的拥塞可视化函数，添加热点检测"""
        # 调用原始函数
        output_files = original_visualize_congestion(
            congestion_data, output_dir, label_postfix)
        
        # 为每个输出文件添加热点分析版本
        hotspot_files = []
        for output_file in output_files:
            # 从文件名推断数据类型
            data_type = ""
            if "horizontal" in output_file:
                data_type = "horizontal"
            elif "vertical" in output_file:
                data_type = "vertical"
            
            # 获取对应的数据矩阵
            if data_type and data_type in congestion_data:
                data_matrix = congestion_data[data_type]['data']
                header = congestion_data[data_type]['header']
                
                # 创建热点可视化
                hotspot_visualizer = HotspotVisualizer(threshold=threshold, min_area=min_area)
                hotspots = hotspot_visualizer.detect_hotspots(data_matrix)
                hotspot_path = hotspot_visualizer.create_hotspot_visualization(
                    data_matrix, output_file, hotspots)
                
                hotspot_files.append(hotspot_path)
        
        # 返回所有输出文件
        return output_files + hotspot_files
    
    # 替换原始函数
    visualizers.visualize_congestion = enhanced_visualize_congestion
    print("已集成热点检测功能到可视化流程")


# 以下代码用于测试和展示
if __name__ == "__main__":
    import sys
    import os
    
    # 如果提供了命令行参数，直接处理指定文件
    if len(sys.argv) > 1:
        from data_parsers import parse_congestion_data
        
        data_file = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
        min_area = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        
        try:
            print(f"解析文件: {data_file}")
            data_matrix, header = parse_congestion_data(data_file)
            
            # 创建输出路径
            output_dir = os.path.dirname(data_file) or '.'
            base_name = os.path.basename(data_file).replace('.data.gz', '')
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            # 创建拥塞图和热点分析图
            original_path, hotspot_path = visualize_with_hotspots(
                data_matrix, output_path, header, threshold, min_area)
            
            print(f"生成文件:\n - {original_path}\n - {hotspot_path}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("用法: python hotspot_visualizer.py <数据文件> [阈值] [最小面积]") 