#!/usr/bin/env python3
"""Run vessel experiments into main/vessel/test_output. Usage from repo root:
  python main/vessel/run_experiments_to_test_output.py
  python main/vessel/run_experiments_to_test_output.py --n_masks 5
"""
import argparse
import os
import sys

if __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _root)
    __package__ = "main.vessel"

from main.vessel.experiment import load_mask, run_experiment_grid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "test_output")
DEFAULT_MASK_DIR = "/media/yuganlab/blackstone/MICCAI2026/datasets/masks/test/Ground truth"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--n_masks", type=int, default=5)
    p.add_argument("--vti_targets", type=float, nargs="+", default=None)
    p.add_argument("--width_scales", type=float, nargs="+", default=None)
    args = p.parse_args()

    mask_dir = os.path.expanduser(args.mask_dir)
    if not os.path.isdir(mask_dir):
        print("Error: not a directory:", mask_dir)
        sys.exit(1)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith(exts) and not f.startswith(".")]
    files.sort()
    to_run = files[: args.n_masks]
    if not to_run:
        print("No masks in", mask_dir)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print("Output:", os.path.abspath(args.out_dir), "| Masks:", to_run)
    for i, fname in enumerate(to_run):
        stem = os.path.splitext(fname)[0]
        path = os.path.join(mask_dir, fname)
        out_sub = os.path.join(args.out_dir, stem)
        print(f"[{i+1}/{len(to_run)}] {fname} -> {out_sub}")
        try:
            mask = load_mask(path)
            run_experiment_grid(mask, out_sub, vti_multipliers=args.vti_targets, width_scales=args.width_scales)
        except Exception as e:
            print("  Failed:", e)
            import traceback
            traceback.print_exc()
    print("Done.")


if __name__ == "__main__":
    main()
