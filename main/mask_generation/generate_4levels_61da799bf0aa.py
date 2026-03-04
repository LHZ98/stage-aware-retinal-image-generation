"""
Generate 4 levels (LV1–LV4) of VTI and mask width from a baseline mask (e.g. 61da799bf0aa).
Baseline is computed at runtime (same as MaskGenerator) so width scales correctly.
"""

import numpy as np
import os
import sys
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.mask_generation import MaskGenerator

# 4 levels: (name, VTI mult, Width mult) — 1.05 / 1.10 / 1.15 / 1.20
LEVELS = [
    ("lv1_mild",      1.05, 1.05),   # +5% VTI, +5% Width
    ("lv2_moderate",  1.10, 1.10),   # +10% VTI, +10% Width
    ("lv3_strong",    1.15, 1.15),   # +15% VTI, +15% Width
    ("lv4_stronger",  1.20, 1.20),   # +20% VTI, +20% Width
]

def main(mask_path_arg=None, use_vessel_style=False, out_dir=None):
    if mask_path_arg and os.path.isfile(mask_path_arg):
        mask_path = os.path.abspath(mask_path_arg)
        case_id = os.path.splitext(os.path.basename(mask_path))[0]
    else:
        mask_candidates = [
            os.path.join(script_dir, "test_output", "61da799bf0aa_original.png"),
            "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP/train_images/train_masks/61da799bf0aa.png",
        ]
        mask_path = None
        for p in mask_candidates:
            if os.path.isfile(p):
                mask_path = p
                break
        if not mask_path:
            print("Mask file not found. Usage: python generate_4levels_61da799bf0aa.py [--mask /path/to/mask.png]")
            return
        case_id = "61da799bf0aa"
    if out_dir is None:
        output_dir = os.path.join(script_dir, "test_output", f"{case_id}_4levels")
    else:
        output_dir = os.path.abspath(out_dir)
    os.makedirs(output_dir, exist_ok=True)

    generator = MaskGenerator()
    baseline = generator.get_baseline_geometry(mask_path)
    base_vti = baseline["VTI"]
    base_width = baseline["Width"]
    print(f"Baseline case: {case_id}")
    print(f"  Baseline VTI:   {base_vti:.6f}")
    print(f"  Baseline Width: {base_width:.6f}")
    print()

    results = []
    for level_name, vti_mult, width_mult in LEVELS:
        target_vti = base_vti * vti_mult
        target_width = base_width * width_mult
        print(f"Generating {level_name}: target VTI={target_vti:.4f}, Width={target_width:.4f}")

        try:
            new_mask, metrics = generator.generate_curved_mask(
                mask_path,
                target_vti=target_vti,
                target_width=target_width,
                use_vessel_style=use_vessel_style,
            )
            suffix = "_vessel" if use_vessel_style else ""
            out_path = os.path.join(output_dir, f"{case_id}_{level_name}{suffix}_generated.png")
            Image.fromarray((new_mask * 255).astype(np.uint8)).save(out_path)
            results.append((level_name, metrics, out_path))
            print(f"  Achieved VTI={metrics['achieved_vti']:.4f}, Width={metrics['achieved_width']:.4f} -> {out_path}")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("4-level summary")
    print("=" * 70)
    for level_name, metrics, out_path in results:
        print(f"  {level_name}: VTI {metrics['achieved_vti']:.4f} (target {metrics['target_vti']:.4f}), "
              f"Width {metrics['achieved_width']:.2f} (target {metrics['target_width']:.2f})")
    print()
    print("Output dir:", output_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate 4-level VTI/Width deformation masks")
    p.add_argument("--mask", type=str, default=None, help="Input mask path (default: 61da799bf0aa)")
    p.add_argument("--vessel_style", action="store_true", help="Use vessel pipeline: path sine + centerline smooth + reconstruct")
    p.add_argument("--out_dir", type=str, default=None, help="Output dir for 4 levels (default: test_output/{case_id}_4levels)")
    args = p.parse_args()
    main(mask_path_arg=args.mask, use_vessel_style=args.vessel_style, out_dir=args.out_dir)
