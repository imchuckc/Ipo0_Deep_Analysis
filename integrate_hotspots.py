#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# integrate_hotspots.py - 将热点检测集成到现有可视化流程

import os
import sys

# 添加当前目录到模块搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入修改模块
from hotspot_visualizer import integrate_with_visualizers

def main():
    """主函数"""
    # 集成热点检测到可视化流程
    integrate_with_visualizers()
    
    print("热点检测功能已成功集成到可视化流程中。")
    print("现在每次生成拥塞图时，将自动生成一个带热点标记的版本。")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 