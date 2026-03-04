"""
Test MaskGenerator.
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

def test_basic_generation():
    """Test basic mask generation."""
    print("=" * 70)
    print("Test MaskGenerator")
    print("=" * 70)

    mask_dir = "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP/train_images/train_masks"
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    if not mask_files:
        print("No mask files found.")
        return

    test_mask_file = os.path.join(mask_dir, mask_files[0])
    case_id = os.path.splitext(os.path.basename(test_mask_file))[0]

    print(f"\nTest mask: {case_id}")
    print(f"Path: {test_mask_file}")

    generator = MaskGenerator()

    print("\n1. Baseline geometry...")
    baseline = generator.get_baseline_geometry(test_mask_file)
    print(f"   Baseline VTI: {baseline['VTI']:.6f}")
    print(f"   Baseline Width: {baseline['Width']:.6f}")
    print(f"   Baseline Density: {baseline['Density']:.6f}")

    target_vti = baseline['VTI'] * 1.15
    target_width = baseline['Width'] * 1.1

    print(f"\n2. Targets:")
    print(f"   Target VTI: {target_vti:.6f} (+{((target_vti/baseline['VTI']-1)*100):.1f}%)")
    print(f"   Target Width: {target_width:.6f} (+{((target_width/baseline['Width']-1)*100):.1f}%)")

    print(f"\n3. Generating mask (precise VTI control)...")
    print("   This may take a while...")
    
    try:
        new_mask, metrics = generator.generate_curved_mask(
            test_mask_file,
            target_vti=target_vti,
            target_width=target_width,
            tolerance=1e-5,
            verify_after_reconstruction=True
        )
        
        print(f"\n4. Results:")
        print(f"   Target VTI: {metrics['target_vti']:.6f}")
        print(f"   Achieved VTI: {metrics['achieved_vti']:.6f}")
        print(f"   VTI error: {metrics['vti_error']:.6f} ({metrics['vti_error']/target_vti*100:.3f}%)")
        print(f"   Target Width: {metrics['target_width']:.6f}")
        print(f"   Achieved Width: {metrics['achieved_width']:.6f}")
        print(f"   Width error: {metrics['width_error']:.6f}")
        print(f"   Main branches: {metrics['main_branches_count']}")
        print(f"   Capillaries: {metrics['capillaries_count']}")

        output_dir = os.path.join(script_dir, "test_output")
        os.makedirs(output_dir, exist_ok=True)

        original_mask = generator.load_mask(test_mask_file)
        Image.fromarray((original_mask * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"{case_id}_original.png")
        )
        
        # Save generated mask
        Image.fromarray((new_mask * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f"{case_id}_generated.png")
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(original_mask, cmap='gray')
        axes[0].set_title(f'Original Mask\nVTI={baseline["VTI"]:.4f}, Width={baseline["Width"]:.2f}',
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(new_mask, cmap='gray')
        axes[1].set_title(f'Generated Mask\nVTI={metrics["achieved_vti"]:.4f}, Width={metrics["achieved_width"]:.2f}',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        diff = new_mask.astype(int) - original_mask.astype(int)
        axes[2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title('Difference\n(Red=added, Blue=removed)', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle(f'Mask generation test - {case_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        viz_path = os.path.join(output_dir, f"{case_id}_comparison.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n5. Saved:")
        print(f"   Original: {os.path.join(output_dir, f'{case_id}_original.png')}")
        print(f"   Generated: {os.path.join(output_dir, f'{case_id}_generated.png')}")
        print(f"   Comparison: {viz_path}")

        print(f"\n6. VTI accuracy:")
        vti_relative_error = abs(metrics['vti_error'] / target_vti) * 100
        if vti_relative_error < 1.0:
            print(f"   VTI error: {vti_relative_error:.3f}% (< 1%) - excellent")
        elif vti_relative_error < 5.0:
            print(f"   VTI error: {vti_relative_error:.3f}% (< 5%) - good")
        else:
            print(f"   VTI error: {vti_relative_error:.3f}% (>= 5%) - needs improvement")

        return True
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_generation()
    if success:
        print("\n" + "=" * 70)
        print("Test complete.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Test failed.")
        print("=" * 70)
