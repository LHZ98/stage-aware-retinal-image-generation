#!/usr/bin/env python3
"""
Visualize nnUNet prediction masks for easier inspection.
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def visualize_prediction(pred_file, output_file=None, overlay_alpha=0.5):
    """
    可视化预测结果
    
    Args:
        pred_file: 预测mask文件路径
        output_file: 输出可视化图像路径（如果为None，会在原文件同目录生成）
        overlay_alpha: 叠加透明度（0-1）
    """
    # 读取预测mask
    mask = np.array(Image.open(pred_file))
    
    # 检查mask值
    unique_vals = np.unique(mask)
    print(f"文件: {os.path.basename(pred_file)}")
    print(f"  唯一值: {unique_vals}")
    print(f"  血管像素占比: {np.sum(mask == 255) / mask.size * 100:.2f}%")
    
    # 创建可视化图像
    # 方法1: 直接显示mask（白色血管，黑色背景）
    vis_image = mask.copy()
    
    # 方法2: 创建彩色可视化（可选）
    # 将0/255转换为RGB图像，血管显示为红色
    if len(vis_image.shape) == 2:
        vis_rgb = np.zeros((*vis_image.shape, 3), dtype=np.uint8)
        vis_rgb[mask == 255] = [255, 0, 0]  # 血管显示为红色
        vis_image = vis_rgb
    
    # 保存可视化结果
    if output_file is None:
        output_file = str(Path(pred_file).parent / f"{Path(pred_file).stem}_vis.png")
    
    Image.fromarray(vis_image).save(output_file)
    print(f"  可视化结果保存到: {output_file}")
    print()


def visualize_all_predictions(pred_dir, output_dir=None):
    """
    批量可视化所有预测结果
    """
    pred_path = Path(pred_dir)
    
    if output_dir is None:
        output_dir = pred_path / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 找到所有PNG文件（排除输入文件）
    png_files = [f for f in pred_path.glob("*.png") 
                 if not f.name.endswith("_0000.png") 
                 and not f.name.endswith("_vis.png")]
    
    print(f"找到 {len(png_files)} 个预测文件")
    print(f"可视化结果将保存到: {output_dir}\n")
    
    for png_file in png_files:
        output_file = output_dir / f"{png_file.stem}_vis.png"
        visualize_prediction(png_file, output_file)
    
    print(f"\n✅ 可视化完成！共处理 {len(png_files)} 个文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化nnUNet预测结果')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='预测结果文件夹路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='可视化结果输出文件夹（默认在输入文件夹下创建visualizations子文件夹）')
    
    args = parser.parse_args()
    
    visualize_all_predictions(args.input_dir, args.output_dir)



