#!/usr/bin/env python3
"""
Process train images and masks with fundus circle cropping; keeps image-mask alignment.
"""

import os
import math
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
    img_id: str,
    output_img_dir: Path,
    output_mask_dir: Path,
    resize_shape: int = 1024,
    resize_canny_edge: int = 1000,
    sigma_scale: int = 50,
    circle_fit_steps: int = 100,
    λ: float = 0.01,
    remove_rectangles: bool = True,
):
    """处理图像和对应的mask"""
    
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
    
    # 保存图像
    if len(cropped_img.shape) == 3:
        Image.fromarray(cropped_img).save(output_img_dir / f"{img_id}.png")
    else:
        Image.fromarray(cropped_img, mode='L').save(output_img_dir / f"{img_id}.png")
    
    # 保存mask
    Image.fromarray(cropped_mask, mode='L').save(output_mask_dir / f"{img_id}.png")
    
    return False  # success


def process_samples():
    """处理前10个样本"""
    
    # 输入路径
    input_img_dir = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/train/Original")
    input_mask_dir = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/train/Ground truth")
    
    # 输出路径（与SYSU样本放在一起）
    output_dir = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/the-sustechsysu-dataset/samples")
    
    images_cropped_dir = output_dir / "images_cropped"
    masks_dir = output_dir / "masks"
    
    # 创建输出目录
    images_cropped_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取前10个图像文件
    image_files = sorted(input_img_dir.glob("*.png"))[:10]
    
    print(f"开始处理 {len(image_files)} 个train样本...")
    print(f"输入图像目录: {input_img_dir}")
    print(f"输入mask目录: {input_mask_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    failures = []
    
    for idx, img_path in enumerate(image_files, 1):
        # 获取对应的mask文件
        mask_path = input_mask_dir / img_path.name
        
        if not mask_path.exists():
            print(f"[{idx}/{len(image_files)}] ⚠️  Mask文件不存在: {mask_path.name}")
            failures.append(img_path.stem)
            continue
        
        img_id = img_path.stem  # 例如 "1_A"
        
        print(f"[{idx}/{len(image_files)}] 处理: {img_path.name} (ID: {img_id})")
        
        try:
            # 读取原始尺寸
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            print(f"  原始图像尺寸: {img.shape}, Mask尺寸: {mask.shape}")
            
            # 处理图像和mask
            failure = process_image_and_mask(
                img_path=img_path,
                mask_path=mask_path,
                img_id=img_id,
                output_img_dir=images_cropped_dir,
                output_mask_dir=masks_dir,
                resize_shape=1024,
                resize_canny_edge=1000,
                sigma_scale=50,
                circle_fit_steps=100,
                λ=0.01,
                remove_rectangles=True,
            )
            
            if failure:
                print(f"  ⚠️  处理失败: {img_path.name}")
                failures.append(img_id)
            else:
                # 检查输出文件
                cropped_img_path = images_cropped_dir / f"{img_id}.png"
                cropped_mask_path = masks_dir / f"{img_id}.png"
                
                if cropped_img_path.exists() and cropped_mask_path.exists():
                    cropped_img = np.array(Image.open(cropped_img_path))
                    cropped_mask = np.array(Image.open(cropped_mask_path))
                    print(f"  ✓ 成功! 裁剪后图像尺寸: {cropped_img.shape}, Mask尺寸: {cropped_mask.shape}")
                else:
                    print(f"  ⚠️  输出文件未找到")
                    failures.append(img_id)
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            failures.append(img_id)
        
        print()
    
    # 保存失败列表
    if failures:
        failure_file = output_dir / "failures_train.lst"
        with open(failure_file, "w") as f:
            f.write("\n".join(failures))
        print(f"⚠️  有 {len(failures)} 个文件处理失败，已保存到: {failure_file}")
    else:
        print("✓ 所有样本处理成功!")
    
    print(f"\n处理完成! 结果保存在: {output_dir}")
    print(f"  - 裁剪后的图像: {images_cropped_dir}")
    print(f"  - 裁剪后的mask: {masks_dir}")


if __name__ == "__main__":
    process_samples()

