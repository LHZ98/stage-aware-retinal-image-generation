#!/usr/bin/env python3
"""
Prepare APTOS2019 data for segmentation-guided diffusion training.
1. Fix mask format (255 -> 1)
2. Reorganize directory layout
3. Resize images to 256x256
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def fix_mask_format(mask_path, output_path):
    """Convert 255 to 1 in mask."""
    mask = np.array(Image.open(mask_path))
    mask = np.where(mask > 127, 1, 0).astype(np.uint8)
    Image.fromarray(mask).save(output_path)

def resize_image(img_path, output_path, target_size=256):
    """Resize image to target size."""
    img = Image.open(img_path)
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_resized.save(output_path)

def prepare_diffusion_data():
    """Prepare data for diffusion training."""

    base_input = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP")
    base_output = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/APTOS2019_DIFFUSION")
    img_output = base_output / "images"
    mask_output = base_output / "masks" / "all"
    target_size = 256

    print("="*60)
    print("Prepare APTOS2019 for Segmentation-Guided Diffusion")
    print("="*60)
    print(f"Input: {base_input}")
    print(f"Output: {base_output}")
    print(f"Target size: {target_size}x{target_size}")
    print()

    for split in ['train', 'val', 'test']:
        (img_output / split).mkdir(parents=True, exist_ok=True)
        (mask_output / split).mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': {
            'img_dir': base_input / "train_images" / "train_images",
            'mask_dir': base_input / "train_images" / "train_masks"
        },
        'val': {
            'img_dir': base_input / "val_images" / "val_images",
            'mask_dir': base_input / "val_images" / "val_masks"
        },
        'test': {
            'img_dir': base_input / "test_images" / "test_images",
            'mask_dir': base_input / "test_images" / "test_masks"
        }
    }
    
    total_processed = 0
    
    for split_name, dirs in splits.items():
        print(f"\nProcessing {split_name}...")
        img_dir = dirs['img_dir']
        mask_dir = dirs['mask_dir']

        if not img_dir.exists() or not mask_dir.exists():
            print(f"  Skip {split_name}: directory missing")
            continue
        
        img_files = sorted(img_dir.glob("*.png"))

        if not img_files:
            print(f"  Skip {split_name}: no image files found")
            continue

        print(f"  Found {len(img_files)} image files")

        success_count = 0
        failed = []

        for img_path in tqdm(img_files, desc=f"  {split_name}"):
            try:
                mask_path = mask_dir / img_path.name

                if not mask_path.exists():
                    print(f"  Warning: mask not found {mask_path.name}")
                    failed.append(img_path.name)
                    continue

                output_img_path = img_output / split_name / img_path.name
                output_mask_path = mask_output / split_name / img_path.name

                resize_image(img_path, output_img_path, target_size)
                mask = np.array(Image.open(mask_path))
                mask = np.where(mask > 127, 1, 0).astype(np.uint8)
                mask_img = Image.fromarray(mask)
                mask_resized = mask_img.resize((target_size, target_size), Image.Resampling.NEAREST)
                mask_resized.save(output_mask_path)
                
                success_count += 1
                total_processed += 1
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                failed.append(img_path.name)

        print(f"  {split_name} done: {success_count}/{len(img_files)} success")
        if failed:
            print(f"  Failed: {len(failed)} files")

    print("\n" + "="*60)
    print("Done.")
    print("="*60)
    print(f"Total processed: {total_processed} image-mask pairs")
    print(f"\nOutput layout:")
    print(f"  {base_output}/")
    print(f"    images/")
    print(f"      train/")
    print(f"      val/")
    print(f"      test/")
    print(f"    masks/")
    print(f"      all/")
    print(f"        train/")
    print(f"        val/")
    print(f"        test/")
    print()
    print("Data ready for training.")

if __name__ == "__main__":
    prepare_diffusion_data()
