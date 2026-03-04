"""
Example usage for MaskGenerator and StageRecommender.
"""

import numpy as np
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.mask_generation import MaskGenerator, StageRecommender


def example_basic_usage():
    """Basic usage example."""
    print("=" * 70)
    print("Example 1: Basic usage - generate mask with target VTI and width")
    print("=" * 70)

    generator = MaskGenerator()

    # Load mask (replace with your path)
    mask_path = "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP/train_images/train_masks"
    print("Note: provide a real mask file path to run.")

    # Get baseline geometry
    # baseline = generator.get_baseline_geometry(mask_path)
    # print(f"Baseline: VTI={baseline['VTI']:.6f}, Width={baseline['Width']:.6f}")

    # Generate more curved mask
    # target_vti = baseline['VTI'] * 1.15
    # target_width = baseline['Width'] * 1.1
    # new_mask, metrics = generator.generate_curved_mask(mask_path, target_vti=target_vti, target_width=target_width)
    # print(f"Target VTI: {target_vti:.6f}, Achieved: {metrics['achieved_vti']:.6f}")
    # print(f"Target Width: {target_width:.6f}, Achieved: {metrics['achieved_width']:.6f}")


def example_stage_recommendation():
    """Stage recommendation example."""
    print("\n" + "=" * 70)
    print("Example 2: Stage recommendation - generate masks for all stages from stage 3")
    print("=" * 70)
    
    recommender = StageRecommender()
    
    # Load stage statistics
    stage_stats = recommender.load_stage_statistics()
    print(f"\nStage statistics:")
    for stage, stats in sorted(stage_stats.items()):
        print(f"  Stage {stage}: VTI={stats['VTI']:.6f}, Width={stats['Width']:.6f}")
    
    # Generate masks for all stages (requires real mask file)
    # mask_path = "path/to/stage3/mask.png"
    # source_stage = 3
    # 
    # all_stages_masks = recommender.generate_all_stages(mask_path, source_stage)
    # 
    # print(f"\nGenerated masks for {len(all_stages_masks)} stages:")
    # for stage, (mask, metrics) in sorted(all_stages_masks.items()):
    #     print(f"  Stage {stage}: VTI={metrics['achieved_vti']:.6f}, "
    #           f"Width={metrics['achieved_width']:.6f}")


def example_custom_parameters():
    """Custom parameters example."""
    print("\n" + "=" * 70)
    print("Example 3: Custom parameters - fine control")
    print("=" * 70)
    
    generator = MaskGenerator()
    
    # Custom parameters
    # mask_path = "path/to/mask.png"
    # 
    # new_mask, metrics = generator.generate_curved_mask(
    #     mask_path,
    #     target_vti=1.15,
    #     target_width=4.0,
    #     main_branch_vti_ratio=0.3,
    #     capillary_vti_ratio=1.7,
    #     main_branch_width_ratio=1.2,
    #     capillary_width_ratio=0.8,
    #     endpoint_perturbation_range=(3.0, 5.0),
    #     tolerance=1e-5,
    #     verify_after_reconstruction=True
    # )
    # 
    # print(f"Done. VTI error: {metrics['vti_error']:.6f}")


if __name__ == "__main__":
    print("Vessel mask generator - usage examples")
    print("=" * 70)
    
    example_basic_usage()
    example_stage_recommendation()
    example_custom_parameters()
    
    print("\n" + "=" * 70)
    print("Note: these examples need real mask files to run.")
    print("Replace the paths in the examples with your mask paths.")
    print("=" * 70)
