#!/usr/bin/env python3
"""
Assess whether APTOS2019 CROP data is suitable for training segmentation-guided diffusion.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

def assess_data_for_diffusion():
    """Assess if data is suitable for diffusion training."""

    base_dir = Path("/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP")

    print("="*60)
    print("Assess APTOS2019 CROP for Segmentation-Guided Diffusion")
    print("="*60)
    print()

    print("1. Checking directory structure...")
    train_img_dir = base_dir / "train_images" / "train_images"
    train_mask_dir = base_dir / "train_images" / "train_masks"
    val_img_dir = base_dir / "val_images" / "val_images"
    val_mask_dir = base_dir / "val_images" / "val_masks"
    test_img_dir = base_dir / "test_images" / "test_images"
    test_mask_dir = base_dir / "test_images" / "test_masks"
    
    dirs_exist = {
        "Train images": train_img_dir.exists(),
        "Train masks": train_mask_dir.exists(),
        "Val images": val_img_dir.exists(),
        "Val masks": val_mask_dir.exists(),
        "Test images": test_img_dir.exists(),
        "Test masks": test_mask_dir.exists(),
    }

    for name, exists in dirs_exist.items():
        status = "OK" if exists else "MISSING"
        print(f"   {status} {name}: {exists}")

    print()

    print("2. Counting files...")
    train_images = list(train_img_dir.glob("*.png")) if train_img_dir.exists() else []
    train_masks = list(train_mask_dir.glob("*.png")) if train_mask_dir.exists() else []
    val_images = list(val_img_dir.glob("*.png")) if val_img_dir.exists() else []
    val_masks = list(val_mask_dir.glob("*.png")) if val_mask_dir.exists() else []
    test_images = list(test_img_dir.glob("*.png")) if test_img_dir.exists() else []
    test_masks = list(test_mask_dir.glob("*.png")) if test_mask_dir.exists() else []
    
    print(f"   Train: {len(train_images)} images, {len(train_masks)} masks")
    print(f"   Val: {len(val_images)} images, {len(val_masks)} masks")
    print(f"   Test: {len(test_images)} images, {len(test_masks)} masks")
    print()

    print("3. Checking image-mask match...")
    if train_images and train_masks:
        img_names = {f.stem for f in train_images}
        mask_names = {f.stem for f in train_masks}
        matched = img_names & mask_names
        unmatched_img = img_names - mask_names
        unmatched_mask = mask_names - img_names
        
        print(f"   Matched image-mask pairs: {len(matched)}")
        if unmatched_img:
            print(f"   Warning: images without mask: {len(unmatched_img)}")
        if unmatched_mask:
            print(f"   Warning: masks without image: {len(unmatched_mask)}")
    print()

    print("4. Checking image format...")
    if train_images:
        sample_img = Image.open(train_images[0])
        print(f"   Image size: {sample_img.size}")
        print(f"   Image mode: {sample_img.mode}")
        print(f"   Image format: {sample_img.format}")
    print()

    print("5. Checking mask format...")
    if train_masks:
        sample_mask = np.array(Image.open(train_masks[0]))
        print(f"   Mask shape: {sample_mask.shape}")
        print(f"   Mask dtype: {sample_mask.dtype}")
        print(f"   Mask unique values: {sorted(np.unique(sample_mask))}")

        # Mask values should be integer 0, 1, 2, ...
        unique_vals = np.unique(sample_mask)
        is_integer = all(val == int(val) for val in unique_vals)
        starts_from_zero = 0 in unique_vals
        is_consecutive = all(i in unique_vals for i in range(len(unique_vals)))
        
        print(f"   Integer labels: {is_integer}")
        print(f"   Starts from 0: {starts_from_zero}")
        print(f"   ✅ 连续标签: {is_consecutive}")
        
        if not (is_integer and starts_from_zero):
            print("   Warning: mask format may not meet requirements (integer 0, 1, 2, ...)")
    print()

    print("6. Assessing data volume...")
    total_train = len(train_images)
    print(f"   Train image count: {total_train}")

    if total_train >= 1000:
        print(f"   Data volume OK ({total_train} >= 1000)")
    elif total_train >= 500:
        print(f"   Moderate volume ({total_train} >= 500); consider more epochs or augmentation")
    else:
        print(f"   Low volume ({total_train} < 500); consider adding more data")
    print()

    print("7. Checking directory layout for segmentation-guided-diffusion...")
    print("   Required layout:")
    print("   DATA_FOLDER/")
    print("     train/")
    print("     val/")
    print("     test/")
    print("   MASK_FOLDER/")
    print("     all/")
    print("       train/")
    print("       val/")
    print("       test/")
    print()
    print("   Current layout:")
    print("   CROP/")
    print("     train_images/train_images/")
    print("     train_images/train_masks/")
    print("     val_images/val_images/")
    print("     val_images/val_masks/")
    print("     test_images/test_images/")
    print("     test_images/test_masks/")
    print()
    print("   Reorganize directory layout to match requirements.")
    print()

    print("="*60)
    print("Summary and recommendations")
    print("="*60)
    print()

    issues = []
    recommendations = []

    if total_train < 1000:
        issues.append("Data volume may be low")
        recommendations.append("Consider augmentation or more data")

    if not (is_integer and starts_from_zero) if train_masks else True:
        issues.append("Mask format may not meet requirements")
        recommendations.append("Ensure masks are integer labels (0, 1, 2, ...), 0 = background")

    if not all(dirs_exist.values()):
        issues.append("Some directories missing")
        recommendations.append("Ensure all train/val/test dirs exist")

    if len(matched) < len(train_images) if train_images and train_masks else False:
        issues.append("Image and mask not fully matched")
        recommendations.append("Ensure each image has a corresponding mask")

    if issues:
        print("Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print()

    if recommendations:
        print("Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        print()

    print("Final assessment:")
    if total_train >= 1000 and all(dirs_exist.values()) and len(matched) == len(train_images) if train_images and train_masks else False:
        print("   Data is suitable for training; still need to:")
        print("      1. Reorganize directory layout")
        print("      2. Verify mask format (integer 0, 1, 2, ...)")
        print("      3. Set image_size (data is 512x512; resize to 256 or keep 512)")
    else:
        print("   Data may need adjustments before training")
    
    print()
    print("="*60)

if __name__ == "__main__":
    assess_data_for_diffusion()
