"""
Test two examples using main/vessel code.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.vessel.experiment import load_mask, run_single_experiment
from main.vessel.skeleton_utils import extract_skeleton_and_segments
from main.vessel.metrics import compute_global_tortuosity, compute_global_width

def test_two_examples():
    """Test mask generation on two examples."""
    print("=" * 70)
    print("Test two examples with main/vessel code")
    print("=" * 70)

    mask_dir = "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP/train_images/train_masks"
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    if len(mask_files) < 2:
        print("Need at least 2 mask files.")
        return

    test_masks = mask_files[:2]
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    for idx, mask_file in enumerate(test_masks):
        case_id = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(mask_dir, mask_file)

        print(f"\n{'='*70}")
        print(f"Test {idx+1}/2: {case_id}")
        print(f"{'='*70}")

        try:
            mask = load_mask(mask_path)
            shape = mask.shape

            print("\n1. Extracting segments...")
            _, _, segments, segment_types = extract_skeleton_and_segments(mask)
            if not segments:
                print("Could not extract segments.")
                continue

            baseline_vti = compute_global_tortuosity(segments)
            baseline_width = compute_global_width(segments)
            print(f"   Baseline VTI: {baseline_vti:.6f}")
            print(f"   Baseline Width: {baseline_width:.6f}")

            target_vti = baseline_vti * 1.15
            alpha = 1.1
            target_width = baseline_width * alpha

            print(f"\n2. Targets:")
            print(f"   Target VTI: {target_vti:.6f} (+{((target_vti/baseline_vti-1)*100):.1f}%)")
            print(f"   Target Width: {target_width:.6f} (+{((alpha-1)*100):.1f}%)")

            print(f"\n3. Generating mask (run_single_experiment)...")
            mask_out, metrics = run_single_experiment(
                segments, segment_types, shape, target_vti, alpha
            )

            print(f"\n4. Results:")
            print(f"   Target VTI: {metrics['target_VTI']:.6f}")
            print(f"   Achieved VTI: {metrics['achieved_VTI']:.6f}")
            print(f"   VTI error: {abs(metrics['achieved_VTI'] - metrics['target_VTI']):.6f} "
                  f"({abs(metrics['achieved_VTI'] - metrics['target_VTI'])/metrics['target_VTI']*100:.3f}%)")
            print(f"   Target Caliber: {metrics['target_caliber']:.6f}")
            print(f"   Achieved Caliber: {metrics['achieved_caliber']:.6f}")
            print(f"   Caliber error: {abs(metrics['achieved_caliber'] - metrics['target_caliber']):.6f}")

            Image.fromarray((mask * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"{case_id}_original.png"))
            Image.fromarray((mask_out * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"{case_id}_generated.png"))

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(mask, cmap='gray')
            axes[0].set_title(f'Original Mask\nVTI={baseline_vti:.4f}, Width={baseline_width:.2f}',
                             fontsize=12, fontweight='bold')
            axes[0].axis('off')
            axes[1].imshow(mask_out, cmap='gray')
            axes[1].set_title(f'Generated Mask\nVTI={metrics["achieved_VTI"]:.4f}, Caliber={metrics["achieved_caliber"]:.2f}',
                             fontsize=12, fontweight='bold')
            axes[1].axis('off')
            diff = mask_out.astype(int) - mask.astype(int)
            axes[2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
            axes[2].set_title('Difference\n(Red=Added, Blue=Removed)', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            plt.suptitle(f'Mask Generation Test - {case_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            viz_path = os.path.join(output_dir, f"{case_id}_comparison.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n5. Saved:")
            print(f"   Original: {os.path.join(output_dir, f'{case_id}_original.png')}")
            print(f"   Generated: {os.path.join(output_dir, f'{case_id}_generated.png')}")
            print(f"   Comparison: {viz_path}")

        except Exception as e:
            print(f"\nError processing {case_id}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)

if __name__ == "__main__":
    test_two_examples()
