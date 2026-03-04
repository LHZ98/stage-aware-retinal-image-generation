#!/usr/bin/env python3
"""
眼底图像圆形裁剪预处理脚本
处理图像和对应的mask，保持对齐，直接替换原文件
"""

import os
import math
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from skimage import feature, transform


def expectation(p: np.ndarray, w: np.ndarray) -> Tuple[float]:
    """Expectation step for finding circle in 2D image."""
    x, y = p
    x_mean = w.T @ x
    y_mean = w.T @ y
    u = x - x_mean
    v = y - y_mean
    S_uu = w.T @ (u**2)
    S_uv = w.T @ (u * v)
    S_vv = w.T @ (v**2)
    S_uuu = w.T @ (u**3)
    S_vvv = w.T @ (v**3)
    S_uvv = w.T @ (u * v**2)
    S_vuu = w.T @ (v * u**2)
    A_inv = np.array([[S_vv, -S_uv], [-S_uv, S_uu]]) / (S_uu * S_vv - S_uv**2)
    B = np.array([S_uuu + S_uvv, S_vvv + S_vuu]) / 2
    u_c, v_c = A_inv @ B
    x_c = u_c + x_mean
    y_c = v_c + y_mean
    alpha = u_c**2 + v_c**2 + S_uu + S_vv
    r = np.sqrt(alpha)
    return (x_c, y_c, r)


def maximization(p: np.ndarray, circle: Tuple[float], λ: float = 1.0) -> np.ndarray:
    """Maximization step for finding circle in 2D image."""
    x, y = p
    x_c, y_c, r = circle
    delta = λ * (np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) - r) ** 2
    delta = delta - delta.min()
    w = np.exp(-delta)
    w /= w.sum()
    return w


def circle_em(p: np.ndarray, circle_init: Tuple[float], num_steps: int = 100, λ: float = 0.1) -> Tuple[float]:
    """Iterative expectation-maximization algorithm to find circle in 2D plane."""
    x_c, y_c, r = circle_init
    for step in range(num_steps):
        w = maximization(p, (x_c, y_c, r), λ / (num_steps - step))
        x_c, y_c, r = expectation(p, w)
    return (x_c, y_c, r)


def get_mask(ratios: List[float], target_resolution: int = 1024):
    """Get mask from radius ratios."""
    Y, X = np.ogrid[:target_resolution, :target_resolution]
    r = target_resolution / 2
    dist_from_center = np.sqrt((X - r) ** 2 + (Y - r) ** 2)
    mask = dist_from_center <= r

    if ratios[0] < 1:
        mask &= Y >= (r - r * ratios[0])
    if ratios[1] < 1:
        mask &= Y <= (r + r * ratios[1])

    if ratios[2] < 1:
        mask &= X >= (r - r * ratios[2])
    if ratios[3] < 1:
        mask &= X <= (r + r * ratios[3])
    return mask


def square_padding(im: np.ndarray, add_pad: int = 100) -> np.ndarray:
    """Set image into the center and pad around."""
    dim_y, dim_x = im.shape[:2]
    dim_larger = max(dim_x, dim_y)
    x_pad = (dim_larger - dim_x) // 2 + add_pad
    y_pad = (dim_larger - dim_y) // 2 + add_pad
    if len(im.shape) == 3:
        return np.pad(im, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)))
    else:
        return np.pad(im, ((y_pad, y_pad), (x_pad, x_pad)))


def find_circle_and_crop_params(
    x: np.ndarray,
    resize_canny_edge: int = 1000,
    sigma_scale: int = 50,
    circle_fit_steps: int = 100,
    λ: float = 0.01,
    fit_largest_contour: bool = False,
):
    """找到圆形参数，返回裁剪参数"""
    is_RGB = len(x.shape) == 3

    # 转换为RGB
    if not is_RGB:
        x = np.asarray(Image.fromarray(x).convert(mode="RGB"))

    # Pad image to square
    x_square_padded = square_padding(x)

    # Detect outer circle
    height, _, _ = x_square_padded.shape
    # Resize image for canny edge detection
    x_resized = np.array(
        Image.fromarray(x_square_padded).resize((resize_canny_edge,) * 2)
    )
    x_gray = x_resized.mean(axis=-1)
    edges = feature.canny(
        x_gray,
        sigma=resize_canny_edge / sigma_scale,
        low_threshold=0.99,
        high_threshold=1,
    )
    
    if np.all(edges == 0):
        return None
    
    # Restore original image size
    edges_resized = np.array(Image.fromarray(edges).resize((height,) * 2))

    if fit_largest_contour:
        contours, _ = cv2.findContours(
            edges_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours_sorted = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        points = contours_sorted[0].squeeze().T
    else:
        points = np.flip(np.array(edges_resized.nonzero()), axis=0)

    x_init = height / 2
    r_init = height / 2 - 100
    circle_init = (x_init, x_init, r_init)
    x_c, y_c, r = circle_em(points, circle_init, circle_fit_steps, λ)

    if math.isnan(x_c) or math.isnan(y_c) or math.isnan(r):
        return None

    # 计算原始图像的padding信息
    h, w = x.shape[:2]
    h_pad = w_pad = x_square_padded.shape[0]
    h_diff = h_pad - h
    w_diff = w_pad - w

    return {
        'x_c': x_c,
        'y_c': y_c,
        'r': r,
        'h_pad': h_pad,
        'w_pad': w_pad,
        'h_diff': h_diff,
        'w_diff': w_diff,
        'h': h,
        'w': w,
    }


def apply_crop_and_resize(img: np.ndarray, params: dict, resize_shape: int = 1024, is_mask: bool = False):
    """应用裁剪和resize到图像或mask"""
    # Pad to square
    img_padded = square_padding(img)
    
    # Crop around circle
    x_c = params['x_c']
    y_c = params['y_c']
    r = params['r']
    
    cropped = img_padded[
        int(y_c - r) : int(y_c + r) + 1, 
        int(x_c - r) : int(x_c + r) + 1
    ]
    
    if cropped.size == 0:
        return None
    
    # Resize
    if is_mask:
        # 对于mask，使用最近邻插值保持二值性
        cropped_resized = transform.resize(
            cropped, (resize_shape,) * 2, 
            anti_aliasing=False, 
            order=0,  # 最近邻
            preserve_range=True
        )
        # 确保是二值的
        cropped_resized = (cropped_resized > 0.5).astype(np.uint8) * 255
    else:
        cropped_resized = transform.resize(
            cropped, (resize_shape,) * 2, 
            anti_aliasing=True
        )
        cropped_resized = (cropped_resized * 255).astype(np.uint8)
    
    return cropped_resized


def process_image_and_mask(
    img_path: Path,
    mask_path: Path,
    resize_shape: int = 1024,
    resize_canny_edge: int = 1000,
    sigma_scale: int = 50,
    circle_fit_steps: int = 100,
    λ: float = 0.01,
    remove_rectangles: bool = True,
):
    """处理图像和对应的mask，直接替换原文件"""
    
    # 读取图像
    img = np.array(Image.open(img_path))
    
    # 找到圆形参数（基于图像）
    params = find_circle_and_crop_params(
        img,
        resize_canny_edge=resize_canny_edge,
        sigma_scale=sigma_scale,
        circle_fit_steps=circle_fit_steps,
        λ=λ,
    )
    
    if params is None:
        return True  # failure
    
    # 处理图像
    cropped_img = apply_crop_and_resize(img, params, resize_shape, is_mask=False)
    if cropped_img is None:
        return True
    
    # 读取并处理mask
    mask = np.array(Image.open(mask_path))
    # 如果mask是RGB，转换为灰度
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] if mask.shape[2] >= 1 else mask.mean(axis=2)
    
    cropped_mask = apply_crop_and_resize(mask, params, resize_shape, is_mask=True)
    if cropped_mask is None:
        return True
    
    # 应用圆形mask（移除背景）
    mask_circle = get_mask([1.0, 1.0, 1.0, 1.0], target_resolution=resize_shape)
    if remove_rectangles:
        cropped_img[~mask_circle] = 0
    
    # 创建临时文件
    temp_img_path = img_path.with_suffix('.tmp.png')
    temp_mask_path = mask_path.with_suffix('.tmp.png')
    
    try:
        # 保存到临时文件
        if len(cropped_img.shape) == 3:
            Image.fromarray(cropped_img).save(temp_img_path)
        else:
            Image.fromarray(cropped_img).save(temp_img_path)
        
        Image.fromarray(cropped_mask).save(temp_mask_path)
        
        # 替换原文件
        shutil.move(str(temp_img_path), str(img_path))
        shutil.move(str(temp_mask_path), str(mask_path))
        
        return False  # success
    except Exception as e:
        # 清理临时文件
        if temp_img_path.exists():
            temp_img_path.unlink()
        if temp_mask_path.exists():
            temp_mask_path.unlink()
        raise e


def process_directory(img_dir: Path, mask_dir: Path, dataset_name: str = ""):
    """处理整个目录的所有图像和mask"""
    
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"图像目录: {img_dir}")
    print(f"Mask目录: {mask_dir}")
    print(f"{'='*60}\n")
    
    # 获取所有图像文件
    image_files = sorted(img_dir.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"⚠️  未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件\n")
    
    failures = []
    success_count = 0
    
    for idx, img_path in enumerate(image_files, 1):
        # 获取对应的mask文件
        mask_path = mask_dir / img_path.name
        
        if not mask_path.exists():
            print(f"[{idx}/{len(image_files)}] ⚠️  Mask文件不存在: {mask_path.name}")
            failures.append(img_path.name)
            continue
        
        print(f"[{idx}/{len(image_files)}] 处理: {img_path.name}", end=" ... ")
        
        try:
            # 读取原始尺寸
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            # 处理图像和mask
            failure = process_image_and_mask(
                img_path=img_path,
                mask_path=mask_path,
                resize_shape=1024,
                resize_canny_edge=1000,
                sigma_scale=50,
                circle_fit_steps=100,
                λ=0.01,
                remove_rectangles=True,
            )
            
            if failure:
                print("⚠️  失败")
                failures.append(img_path.name)
            else:
                # 验证结果
                cropped_img = np.array(Image.open(img_path))
                cropped_mask = np.array(Image.open(mask_path))
                print(f"✓ 成功 (图像: {cropped_img.shape}, Mask: {cropped_mask.shape})")
                success_count += 1
            
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            failures.append(img_path.name)
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  成功: {success_count}/{len(image_files)}")
    print(f"  失败: {len(failures)}/{len(image_files)}")
    
    if failures:
        failure_file = img_dir.parent / f"failures_{dataset_name}.lst"
        with open(failure_file, "w") as f:
            f.write("\n".join(failures))
        print(f"  失败列表已保存到: {failure_file}")
    print(f"{'='*60}\n")


def main():
    """主函数：处理train和test目录"""
    
    base_dir = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks")
    
    # 处理train目录
    train_img_dir = base_dir / "train" / "Original"
    train_mask_dir = base_dir / "train" / "Ground truth"
    
    if train_img_dir.exists() and train_mask_dir.exists():
        process_directory(train_img_dir, train_mask_dir, "train")
    else:
        print(f"⚠️  Train目录不存在: {train_img_dir} 或 {train_mask_dir}")
    
    # 处理test目录
    test_img_dir = base_dir / "test" / "Original"
    test_mask_dir = base_dir / "test" / "Ground truth"
    
    if test_img_dir.exists() and test_mask_dir.exists():
        process_directory(test_img_dir, test_mask_dir, "test")
    else:
        print(f"⚠️  Test目录不存在: {test_img_dir} 或 {test_mask_dir}")
    
    print("\n所有处理完成!")


if __name__ == "__main__":
    main()

