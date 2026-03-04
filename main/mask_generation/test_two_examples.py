"""
Test mask generation on two examples.
"""

import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.mask_generation import MaskGenerator

def test_two_examples():
    """Test mask generation on two examples."""
    print("=" * 70)
    print("Test mask generation (two examples)")
    print("=" * 70)

    mask_dir = "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP/train_images/train_masks"
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    if len(mask_files) < 2:
        print("Need at least 2 mask files.")
        return

    test_masks = mask_files[:2]
    generator = MaskGenerator()
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    for idx, mask_file in enumerate(test_masks):
        case_id = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(mask_dir, mask_file)

        print(f"\n{'='*70}")
        print(f"Test {idx+1}/2: {case_id}")
        print(f"{'='*70}")

        try:
            print("\n1. Baseline geometry...")
            baseline = generator.get_baseline_geometry(mask_path)
            print(f"   Baseline VTI: {baseline['VTI']:.6f}")
            print(f"   Baseline Width: {baseline['Width']:.6f}")
            print(f"   Baseline Density: {baseline['Density']:.6f}")

            target_vti = baseline['VTI'] * 1.15
            target_width = baseline['Width'] * 1.1

            print(f"\n2. Targets:")
            print(f"   Target VTI: {target_vti:.6f} (+{((target_vti/baseline['VTI']-1)*100):.1f}%)")
            print(f"   Target Width: {target_width:.6f} (+{((target_width/baseline['Width']-1)*100):.1f}%)")

            print(f"\n3. Generating mask...")
            new_mask, metrics = generator.generate_curved_mask(
                mask_path,
                target_vti=target_vti,
                target_width=target_width
            )

            print(f"\n4. Results:")
            print(f"   Target VTI: {metrics['target_vti']:.6f}")
            print(f"   Achieved VTI: {metrics['achieved_vti']:.6f}")
            print(f"   VTI error: {metrics['vti_error']:.6f} ({metrics['vti_error']/target_vti*100:.3f}%)")
            print(f"   Target Width: {metrics['target_width']:.6f}")
            print(f"   Achieved Width: {metrics['achieved_width']:.6f}")
            print(f"   Width error: {metrics['width_error']:.6f}")

            original_mask = generator.load_mask(mask_path)
            Image.fromarray((original_mask * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"{case_id}_original.png")
            )
            
            Image.fromarray((new_mask * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"{case_id}_generated.png")
            )
            
            # Comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(original_mask, cmap='gray')
            axes[0].set_title(f'Original Mask\nVTI={baseline["VTI"]:.4f}, Width={baseline["Width"]:.2f}', 
                             fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(new_mask, cmap='gray')
            axes[1].set_title(f'Generated Mask\nVTI={metrics["achieved_vti"]:.4f}, Width={metrics["achieved_width"]:.2f}', 
                             fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # Difference map
            diff = new_mask.astype(int) - original_mask.astype(int)
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
