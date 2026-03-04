#!/usr/bin/env python3
"""
Convert 2D retinal fundus images and masks to nnUNet format.

Usage:
    python convert_retinal_fundus_to_nnunet.py \
        --images_dir /path/to/images \
        --masks_dir /path/to/masks \
        --dataset_id 001 \
        --dataset_name RetinalFundus \
        --output_dir /path/to/nnUNet_raw

Or edit paths in the script and run:
    python convert_retinal_fundus_to_nnunet.py
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from skimage import io
from batchgenerators.utilities.file_and_folder_operations import (
    join, maybe_mkdir_p, subfiles, save_json
)
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def find_matching_files(images_dir: str, masks_dir: str, 
                       image_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif'],
                       mask_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif']) -> List[Tuple[str, str]]:
    """
    Find matching image-mask pairs.
    
    Returns:
        List of tuples: [(image_path, mask_path), ...]
    """
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(subfiles(images_dir, suffix=ext, join=False))
    
    # 获取所有mask文件
    mask_files = []
    for ext in mask_extensions:
        mask_files.extend(subfiles(masks_dir, suffix=ext, join=False))
    
    # 创建文件名到完整路径的映射（去除扩展名）
    image_dict = {}
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        image_dict[base_name] = join(images_dir, img_file)
    
    mask_dict = {}
    for mask_file in mask_files:
        base_name = os.path.splitext(mask_file)[0]
        # 处理mask文件名可能包含额外后缀的情况（如 _mask, _gt等）
        # 也尝试原始文件名
        mask_dict[base_name] = join(masks_dir, mask_file)
        # 如果mask文件名包含图像文件名，也添加映射
        for img_base in image_dict.keys():
            if img_base in base_name or base_name in img_base:
                mask_dict[img_base] = join(masks_dir, mask_file)
    
    # 匹配图像和mask
    matched_pairs = []
    for base_name, img_path in image_dict.items():
        # 尝试多种匹配方式
        mask_path = None
        if base_name in mask_dict:
            mask_path = mask_dict[base_name]
        elif f"{base_name}_mask" in mask_dict:
            mask_path = mask_dict[f"{base_name}_mask"]
        elif f"{base_name}_gt" in mask_dict:
            mask_path = mask_dict[f"{base_name}_gt"]
        elif f"{base_name}_label" in mask_dict:
            mask_path = mask_dict[f"{base_name}_label"]
        else:
            # 尝试反向匹配：mask文件名包含图像文件名
            for mask_base, m_path in mask_dict.items():
                if base_name in mask_base or mask_base in base_name:
                    mask_path = m_path
                    break
            
            # 如果还没匹配上，尝试提取数字前缀进行匹配（用于DRIVE等数据集）
            if mask_path is None:
                import re
                # 提取图像文件名中的数字前缀（如 "21" from "21_training"）
                img_numbers = re.findall(r'\d+', base_name)
                if img_numbers:
                    for mask_base, m_path in mask_dict.items():
                        mask_numbers = re.findall(r'\d+', mask_base)
                        # 如果数字前缀匹配，则认为匹配
                        if mask_numbers and img_numbers[0] == mask_numbers[0]:
                            mask_path = m_path
                            break
        
        if mask_path and os.path.exists(mask_path):
            matched_pairs.append((img_path, mask_path))
        else:
            print(f"Warning: no mask for {base_name}")
    
    print(f"Found {len(matched_pairs)} image-mask pairs")
    return matched_pairs


def analyze_mask_labels(mask_paths: List[str]) -> Dict[str, int]:
    """
    Analyze mask labels and build labels dict (binary: background + foreground).
    
    Returns:
        Dict mapping label names to integer values
    """
    all_labels = set()
    
    print("Analyzing mask labels...")
    for mask_path in mask_paths:
        try:
            mask = io.imread(mask_path)
            # 如果是RGB图像，转换为灰度
            if len(mask.shape) == 3:
                # 尝试提取第一个通道或转换为灰度
                mask = mask[:, :, 0] if mask.shape[2] >= 1 else mask
            
            unique_labels = np.unique(mask)
            all_labels.update(unique_labels)
        except Exception as e:
            print(f"Warning: cannot read {mask_path}: {e}")
    
    all_labels = sorted(list(all_labels))
    print(f"Detected labels: {all_labels}")
    
    # 生成labels字典
    labels = {}
    # 背景必须是0
    labels['background'] = 0
    
    # 移除0，处理其他标签
    if 0 in all_labels:
        all_labels.remove(0)
    
    # 对于二分类任务，将所有非0标签合并为一个前景类别
    if len(all_labels) > 0:
        # 如果只有一个非0标签，直接使用
        if len(all_labels) == 1:
            labels['vessel'] = 1  # 或 'foreground'
        else:
            # 如果有多个非0标签，合并为一个前景类别
            print(f"Multiple foreground labels {all_labels}, merging to one class")
            labels['vessel'] = 1  # 或 'foreground'
    else:
        # 如果没有非0标签，仍然添加前景类别（虽然可能不会出现）
        labels['vessel'] = 1
        print("Warning: only background in mask, adding foreground class")
    
    print(f"Labels: {labels} ({len(labels)} classes, binary)")
    return labels


def convert_image_and_mask(image_path: str, mask_path: str, 
                          output_image_path: str, output_mask_path: str,
                          target_image_ext: str = '.png',
                          target_size: int = None):
    """
    转换单个图像和mask到nnUNet格式
    
    Args:
        target_size: 目标尺寸（如果指定，会将图像和mask都resize到此尺寸）
    """
    try:
        # 读取图像
        img = Image.open(image_path)
        # 如果是RGBA，转换为RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # 如果指定了target_size，resize图像
        if target_size is not None:
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # 保存为PNG（无损）
        img.save(output_image_path, 'PNG')
        
        # 读取并处理mask
        # 优先使用PIL读取，因为它对GIF等格式处理更稳定
        try:
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
        except:
            # 如果PIL失败，使用skimage
            mask = io.imread(mask_path)
        
        # 确保mask是2D数组
        if len(mask.shape) == 3:
            # 如果是RGB mask，提取第一个通道
            if mask.shape[2] == 3 or mask.shape[2] == 4:
                mask = mask[:, :, 0]
            else:
                # 如果是其他3D形状，取第一个切片
                mask = mask[0, :, :] if mask.shape[0] < mask.shape[2] else mask[:, :, 0]
        elif len(mask.shape) > 2:
            # 处理其他维度情况
            mask = mask.squeeze()
        
        # 确保mask是2D
        if len(mask.shape) != 2:
            raise ValueError(f"Mask shape invalid: {mask.shape}, expected 2D")
        
        # 确保mask是整数类型
        mask = mask.astype(np.uint8)
        
        # 对于二分类任务，将所有非0值映射为1
        unique_labels = np.unique(mask)
        if len(unique_labels) > 0:
            # 检查是否有非0值
            non_zero_labels = unique_labels[unique_labels != 0]
            if len(non_zero_labels) > 0:
                # 对于二分类，将所有非0值统一映射为1
                mask_binary = np.zeros_like(mask)
                mask_binary[mask != 0] = 1
                mask = mask_binary
                if len(non_zero_labels) > 1:
                    print(f"Multiple foreground labels {non_zero_labels}, mapping to 1")
        
        # 如果指定了target_size，resize mask（使用最近邻插值保持二值性）
        if target_size is not None:
            mask_img = Image.fromarray(mask, mode='L')
            mask_img = mask_img.resize((target_size, target_size), Image.Resampling.NEAREST)
            mask = np.array(mask_img)
        
        # 使用PIL保存mask，确保形状正确
        mask_img = Image.fromarray(mask, mode='L')  # 'L'表示灰度图
        mask_img.save(output_mask_path, 'PNG')
        
    except Exception as e:
        print(f"Error processing {image_path} and {mask_path}: {e}")
        raise


def convert_dataset(images_dir: str, masks_dir: str, 
                   dataset_id: str, dataset_name: str,
                   output_base_dir: str = None,
                   file_ending: str = '.png',
                   target_size: int = None):
    """
    将视网膜眼底图像数据集转换为nnUNet格式
    
    Args:
        images_dir: 图像文件夹路径
        masks_dir: mask文件夹路径
        dataset_id: 数据集ID（3位数字，如 '001'）
        dataset_name: 数据集名称（如 'RetinalFundus'）
        output_base_dir: nnUNet_raw的基础路径（如果为None，将使用环境变量）
        file_ending: 输出文件扩展名（默认.png）
    """
    # 设置输出路径
    if output_base_dir is None:
        from nnunetv2.paths import nnUNet_raw
        output_base_dir = nnUNet_raw
    
    dataset_folder_name = f"Dataset{dataset_id}_{dataset_name}"
    output_dir = join(output_base_dir, dataset_folder_name)
    
    images_tr_dir = join(output_dir, 'imagesTr')
    labels_tr_dir = join(output_dir, 'labelsTr')
    
    maybe_mkdir_p(images_tr_dir)
    maybe_mkdir_p(labels_tr_dir)
    
    print(f"输出目录: {output_dir}")
    
    # 找到匹配的图像和mask文件
    matched_pairs = find_matching_files(images_dir, masks_dir)
    
    if len(matched_pairs) == 0:
        raise ValueError("No matching image-mask pairs. Check paths and filenames.")
    
    # 分析mask标签
    mask_paths = [pair[1] for pair in matched_pairs]
    labels_dict = analyze_mask_labels(mask_paths)
    
    # 转换每个图像和mask对
    print(f"\nConverting {len(matched_pairs)} files...")
    for idx, (image_path, mask_path) in enumerate(matched_pairs):
        # 生成case identifier（使用文件名，去除扩展名）
        case_id = os.path.splitext(os.path.basename(image_path))[0]
        # 清理case_id，移除可能的下划线后缀
        case_id = case_id.replace('_mask', '').replace('_gt', '').replace('_label', '')
        
        # nnUNet格式：图像命名为 {case_id}_0000.{ext}，mask命名为 {case_id}.{ext}
        output_image_path = join(images_tr_dir, f"{case_id}_0000{file_ending}")
        output_mask_path = join(labels_tr_dir, f"{case_id}{file_ending}")
        
        convert_image_and_mask(image_path, mask_path, output_image_path, output_mask_path, file_ending, target_size)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(matched_pairs)}...")
    
    print(f"Converted {len(matched_pairs)} files.")

    print("\nGenerating dataset.json...")
    channel_names = {"0": "R", "1": "G", "2": "B"}  # RGB图像
    
    generate_dataset_json(
        output_folder=output_dir,  # 修复：参数名应该是output_folder
        channel_names=channel_names,
        labels=labels_dict,
        num_training_cases=len(matched_pairs),
        file_ending=file_ending,
        dataset_name=dataset_folder_name,
        overwrite_image_reader_writer="NaturalImage2DIO",
        description="Retinal fundus image segmentation dataset",
        converted_by="nnUNet conversion script"
    )
    
    print(f"\nDone. Output: {output_dir}")
    print("Next: set nnUNet_* env vars, then run nnUNetv2_plan_and_preprocess, nnUNetv2_train.")
    print(f"   export nnUNet_raw=\"{output_base_dir}\"")
    print(f"   export nnUNet_preprocessed=\"/path/to/nnUNet_preprocessed\"")
    print(f"   export nnUNet_results=\"/path/to/nnUNet_results\"")
    print(f"   nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
    print(f"   nnUNetv2_train {dataset_id} 2d 0 --npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert 2D retinal fundus to nnUNet format')
    parser.add_argument('--images_dir', type=str, required=True, help='Image folder')
    parser.add_argument('--masks_dir', type=str, required=True, help='Mask folder')
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset ID (e.g. 001)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--output_dir', type=str, default=None, help='nnUNet_raw path')
    parser.add_argument('--file_ending', type=str, default='.png', help='Output extension')
    parser.add_argument('--target_size', type=int, default=None, help='Resize images/masks to this size')
    args = parser.parse_args()

    if not os.path.exists(args.images_dir):
        raise ValueError(f"Images dir not found: {args.images_dir}")
    if not os.path.exists(args.masks_dir):
        raise ValueError(f"Masks dir not found: {args.masks_dir}")
    convert_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        output_base_dir=args.output_dir,
        file_ending=args.file_ending,
        target_size=args.target_size
    )

