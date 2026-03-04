"""
Run experiment on multiple masks in a directory; each mask gets its own output subdir.

Example (from repo root):
  python main/vessel/run_batch.py \\
    --mask_dir "/path/to/masks" \\
    --n_masks 10 \\
    --out_dir ./vessel_batch_out
"""

import argparse
import os
import sys

# Allow running from repo root or vessel dir
if __name__ == "__main__" and __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "main.vessel"

from main.vessel.experiment import load_mask, run_experiment_grid


def main():
    parser = argparse.ArgumentParser(description="Batch run vessel experiment on multiple masks")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/media/yuganlab/blackstone/MICCAI2026/datasets/masks/test/Ground truth",
        help="Directory containing mask images (.png etc.)",
    )
    parser.add_argument(
        "--n_masks",
        type=int,
        default=10,
        help="Number of masks to process (first N after sorting by filename)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./vessel_batch_out",
        help="Base output directory; each mask gets a subdir named by stem (e.g. 1_A)",
    )
    parser.add_argument(
        "--vti_targets",
        type=float,
        nargs="+",
        default=None,
        help="VTI multipliers, e.g. 1.0 1.05 1.10",
    )
    parser.add_argument(
        "--width_scales",
        type=float,
        nargs="+",
        default=None,
        help="Width scales alpha, e.g. 0.8 1.0 1.2",
    )
    args = parser.parse_args()

    mask_dir = os.path.expanduser(args.mask_dir)
    if not os.path.isdir(mask_dir):
        print("Error: not a directory:", mask_dir)
        sys.exit(1)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
    files = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(exts) and not f.startswith(".")
    ]
    files.sort()
    to_run = files[: args.n_masks]
    if not to_run:
        print("No mask files found in", mask_dir)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print("Processing", len(to_run), "masks:", to_run)

    for i, fname in enumerate(to_run):
        stem = os.path.splitext(fname)[0]
        path = os.path.join(mask_dir, fname)
        out_sub = os.path.join(args.out_dir, stem)
        print(f"[{i+1}/{len(to_run)}] {fname} -> {out_sub}")
        try:
            mask = load_mask(path)
            run_experiment_grid(
                mask,
                out_sub,
                vti_multipliers=args.vti_targets,
                width_scales=args.width_scales,
            )
        except Exception as e:
            print("  Failed:", e)
            import traceback
            traceback.print_exc()

    print("Done. Output under", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
