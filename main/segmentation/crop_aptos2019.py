#!/usr/bin/env python3
"""
Apply circle cropping preprocessing to APTOS2019 dataset (images only, no masks).
"""

import os
import math
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import cv2
from PIL import Image
from skimage import feature, transform, io
from tqdm import tqdm

# --- Circle cropping functions ---
def expectation(p: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
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
    
    det_A = S_uu * S_vv - S_uv**2
    if det_A == 0:
        return float('nan'), float('nan'), float('nan')
    
    A_inv = np.array([[S_vv, -S_uv], [-S_uv, S_uu]]) / det_A
    B = np.array([S_uuu + S_uvv, S_vvv + S_vuu]) / 2
    u_c, v_c = A_inv @ B
    x_c = u_c + x_mean
    y_c = v_c + y_mean
    alpha = u_c**2 + v_c**2 + S_uu + S_vv
    r = np.sqrt(alpha) if alpha >= 0 else float('nan')
    return (x_c, y_c, r)

def maximization(p: np.ndarray, circle: Tuple[float, float, float], λ: float = 1.0) -> np.ndarray:
    x, y = p
    x_c, y_c, r = circle
    
    if any(np.isnan([x_c, y_c, r])):
        return np.ones(len(x)) / len(x)
    
    dist_sq = (x - x_c) ** 2 + (y - y_c) ** 2
    if r <= 0:
        delta = λ * dist_sq
    else:
        delta = λ * (np.sqrt(dist_sq) - r) ** 2
    
    delta = delta - delta.min()
    w = np.exp(-delta)
    w_sum = w.sum()
    if w_sum == 0:
        return np.ones(len(x)) / len(x)
    w /= w_sum
    return w

def circle_em(p: np.ndarray, circle_init: Tuple[float, float, float], num_steps: int = 100, λ: float = 0.1) -> Tuple[float, float, float]:
    x_c, y_c, r = circle_init
    for step in range(num_steps):
        w = maximization(p, (x_c, y_c, r), λ / (num_steps - step))
        x_c, y_c, r = expectation(p, w)
        if any(np.isnan([x_c, y_c, r])):
            break
    return (x_c, y_c, r)

def get_mask(ratios: list, target_resolution: int = 512):
    Y, X = np.ogrid[:target_resolution, :target_resolution]
    r = target_resolution / 2
    dist_from_center = np.sqrt((X - r) ** 2 + (Y - r) ** 2)
    mask = dist_from_center <= r
    return mask

def square_padding(im: np.ndarray, add_pad: int = 100) -> np.ndarray:
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
) -> Optional[Dict]:
    is_RGB = len(x.shape) == 3
    
    if not is_RGB:
        x = np.asarray(Image.fromarray(x).convert(mode="RGB"))
    
    x_square_padded = square_padding(x)
    
    height, _, _ = x_square_padded.shape
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
    
    edges_resized = np.array(Image.fromarray(edges).resize((height,) * 2))
    
    if fit_largest_contour:
        contours, _ = cv2.findContours(
            edges_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        contours_sorted = sorted(contours, key=lambda c: c.shape[0], reverse=True)
        points = contours_sorted[0].squeeze().T
    else:
        points = np.flip(np.array(edges_resized.nonzero()), axis=0)
    
    if points.size == 0:
        return None
    
    x_init = height / 2
    r_init = height / 2 - 100
    circle_init = (x_init, x_init, r_init)
    x_c, y_c, r = circle_em(points, circle_init, circle_fit_steps, λ)
    
    if math.isnan(x_c) or math.isnan(y_c) or math.isnan(r) or r <= 0:
        return None
    
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

def apply_crop_and_resize(img: np.ndarray, params: Dict, resize_shape: int = 512, is_mask: bool = False):
    img_padded = square_padding(img)
    
    x_c = params['x_c']
    y_c = params['y_c']
    r = params['r']
    
    y_start = max(0, int(y_c - r))
    y_end = min(img_padded.shape[0], int(y_c + r) + 1)
    x_start = max(0, int(x_c - r))
    x_end = min(img_padded.shape[1], int(x_c + r) + 1)
    
    cropped = img_padded[y_start:y_end, x_start:x_end]
    
    if cropped.size == 0:
        return None
    
    if is_mask:
        cropped_resized = transform.resize(
            cropped, (resize_shape,) * 2, 
            anti_aliasing=False, 
            order=0,
            preserve_range=True
        )
        cropped_resized = (cropped_resized > 0.5).astype(np.uint8) * 255
    else:
        cropped_resized = transform.resize(
            cropped, (resize_shape,) * 2, 
            anti_aliasing=True,
            preserve_range=True
        )
        if img.dtype == np.uint8:
            cropped_resized = np.clip(cropped_resized, 0, 255).astype(np.uint8)
    
    return cropped_resized

def process_single_image(
    img_path: Path,
    output_img_path: Path,
    resize_shape: int = 512,
    resize_canny_edge: int = 1000,
    sigma_scale: int = 50,
    circle_fit_steps: int = 100,
    λ: float = 0.01,
    remove_rectangles: bool = True,
) -> bool:
    """Process a single image."""
    try:
        img = np.array(Image.open(img_path))
        
        params = find_circle_and_crop_params(
            img,
            resize_canny_edge=resize_canny_edge,
            sigma_scale=sigma_scale,
            circle_fit_steps=circle_fit_steps,
            λ=λ,
        )
        
        if params is None:
            return True  # failure
        
        cropped_img = apply_crop_and_resize(img, params, resize_shape, is_mask=False)
        if cropped_img is None:
            return True
        
        mask_circle = get_mask([1.0, 1.0, 1.0, 1.0], target_resolution=resize_shape)
        if remove_rectangles:
            if len(cropped_img.shape) == 3:
                for c in range(cropped_img.shape[2]):
                    cropped_img[:, :, c][~mask_circle] = 0
            else:
                cropped_img[~mask_circle] = 0
        
        # Save processed image
        if len(cropped_img.shape) == 3:
            Image.fromarray(cropped_img.astype(np.uint8)).save(output_img_path)
        else:
            Image.fromarray(cropped_img.astype(np.uint8)).save(output_img_path)
        
        return False  # success
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return True

def process_directory(
    input_dir: Path,
    output_dir: Path,
    dataset_type: str,
    target_size: int = 512,
    image_extensions: list = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
):
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_type}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found for {dataset_type}, skipping.")
        return
    
    print(f"✅ 找到 {len(image_files)} 个图像文件\n")
    
    success_count = 0
    failures = []
    
    for img_path in tqdm(image_files, desc=f"Processing {dataset_type}"):
        output_img_path = output_dir / img_path.name
        
        failure = process_single_image(
            img_path=img_path,
            output_img_path=output_img_path,
            resize_shape=target_size
        )
        
        if failure:
            failures.append(img_path.name)
        else:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Done: {dataset_type}")
    print(f"  Success: {success_count}/{len(image_files)}")
    print(f"  Failed: {len(failures)}/{len(image_files)}")
    if failures:
        print(f"  Failed files: {', '.join(failures[:10])}{'...' if len(failures) > 10 else ''}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    base_input_dir = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/aptos2019")
    base_output_dir = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP")
    target_fixed_size = 512
    
    train_input_dir = base_input_dir / "train_images" / "train_images"
    train_output_dir = base_output_dir / "train_images" / "train_images"
    if train_input_dir.exists():
        process_directory(
            train_input_dir,
            train_output_dir,
            "train",
            target_size=target_fixed_size
        )

    test_input_dir = base_input_dir / "test_images" / "test_images"
    test_output_dir = base_output_dir / "test_images" / "test_images"
    if test_input_dir.exists():
        process_directory(
            test_input_dir,
            test_output_dir,
            "test",
            target_size=target_fixed_size
        )

    val_input_dir = base_input_dir / "val_images" / "val_images"
    val_output_dir = base_output_dir / "val_images" / "val_images"
    if val_input_dir.exists():
        process_directory(
            val_input_dir,
            val_output_dir,
            "val",
            target_size=target_fixed_size
        )

    print("All processing done.")
