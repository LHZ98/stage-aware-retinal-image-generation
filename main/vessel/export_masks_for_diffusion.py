#!/usr/bin/env python3
"""
Export vessel .npy masks to 256x256 PNG for segmentation-guided diffusion.
Usage (from repo root):
  python main/vessel/export_masks_for_diffusion.py \
    --mask_root main/vessel/test_output/progressive_4levels_5samples \
    --out_dir main/vessel/test_output/progressive_4levels_masks_256
"""
import os
import argparse
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_root", type=str, default="main/vessel/test_output/progressive_4levels_5samples",
                  help="Root dir with subdirs (109_G, 80_D, ...) each containing mask_*.npy")
    p.add_argument("--out_dir", type=str, default="main/vessel/test_output/progressive_4levels_masks_256",
                  help="Output dir for flat list of 256x256 PNG masks")
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    mask_root = os.path.abspath(args.mask_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for sub in sorted(os.listdir(mask_root)):
        subpath = os.path.join(mask_root, sub)
        if not os.path.isdir(subpath):
            continue
        for f in sorted(os.listdir(subpath)):
            if not f.startswith("mask_") or not f.endswith(".npy"):
                continue
            path = os.path.join(subpath, f)
            arr = np.load(path)
            if arr.ndim == 3:
                arr = arr.squeeze()
            if arr.max() <= 1:
                arr = (arr > 0.5).astype(np.uint8) * 255
            else:
                arr = (arr > 0).astype(np.uint8) * 255
            pil = Image.fromarray(arr).convert("L")
            pil = pil.resize((args.size, args.size), Image.NEAREST)
            # name: 109_G_path_sinA3_0.png (case_id + mask stem without "mask_")
            stem = f.replace("mask_", "").replace(".npy", "")
            out_name = f"{sub}_{stem}.png"
            out_path = os.path.join(out_dir, out_name)
            pil.save(out_path)
            count += 1
    print("Exported", count, "masks to", out_dir)


if __name__ == "__main__":
    main()
