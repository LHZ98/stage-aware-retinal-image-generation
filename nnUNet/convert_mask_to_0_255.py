#!/usr/bin/env python3
"""
Convert nnUNet binary masks (0/1) to 0/255 format.
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        return iterable


def convert_mask_to_0_255(input_dir, output_dir=None, inplace=False):
    """
    Convert mask files from 0/1 to 0/255.
    input_dir: folder with PNG masks.
    output_dir: if None and not inplace, creates input_dir_0_255.
    inplace: overwrite original files.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Input dir does not exist: {input_dir}")
    if inplace:
        output_path = input_path
    else:
        if output_dir is None:
            output_path = input_path.parent / f"{input_path.name}_0_255"
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    png_files = [f for f in input_path.glob("*.png")
                 if not f.name.endswith("_0000.png")]

    if len(png_files) == 0:
        print(f"No PNG files in {input_dir}")
        return

    print(f"Found {len(png_files)} PNG files")
    print(f"Output: {output_path}")
    print("Mode: inplace" if inplace else "Mode: new folder")
    print()

    converted_count = 0
    for png_file in tqdm(png_files, desc="Converting"):
        try:
            mask = np.array(Image.open(png_file))
            unique_vals = np.unique(mask)
            if not (set(unique_vals).issubset({0, 1})):
                if set(unique_vals).issubset({0, 255}):
                    continue
                print(f"Skip {png_file.name}: not binary (values {unique_vals})")
                continue

            mask_0_255 = (mask * 255).astype(np.uint8)
            if inplace:
                output_file = png_file
            else:
                output_file = output_path / png_file.name
            
            Image.fromarray(mask_0_255).save(output_file, 'PNG')
            converted_count += 1
            
        except Exception as e:
            print(f"Error processing {png_file.name}: {e}")

    print(f"\nDone: {converted_count}/{len(png_files)} files")
    if not inplace:
        print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert binary mask 0/1 to 0/255')
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder with PNG masks')
    parser.add_argument('--output_dir', type=str, default=None, help='Output folder (optional)')
    parser.add_argument('--inplace', action='store_true', help='Overwrite in place')
    
    args = parser.parse_args()
    
    convert_mask_to_0_255(args.input_dir, args.output_dir, args.inplace)
