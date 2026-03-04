"""
Stage recommender: recommend and generate masks for other stages from a source stage.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional
from collections import defaultdict

from .mask_generator import MaskGenerator


class StageRecommender:
    """
    Recommend and generate masks for other stages from a given stage mask.
    """

    def __init__(
        self,
        mask_generator: Optional[MaskGenerator] = None,
        stats_file: Optional[str] = None
    ):
        """
        Args:
            mask_generator: MaskGenerator instance (creates new if None)
            stats_file: path to stage stats JSON (if None, computed from data)
        """
        self.mask_generator = mask_generator or MaskGenerator()
        self.stats_file = stats_file
        self.stage_stats = None

        # Data paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        self.base_dir = "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/aptos2019"
        self.geometry_data_file = os.path.join(
            os.path.dirname(script_dir), "geometry_case_data.json"
        )
    
    def load_stage_statistics(self) -> Dict:
        """
        Load or compute mean VTI and width per stage (0-4).
        Returns: {stage: {"VTI": mean_vti, "Width": mean_width}}
        """
        if self.stage_stats is not None:
            return self.stage_stats

        # Load from stats_file if provided
        if self.stats_file and os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                data = json.load(f)
                self.stage_stats = {}
                for stage_str, stats in data.items():
                    self.stage_stats[int(stage_str)] = {
                        "VTI": stats["VTI"]["mean"],
                        "Width": stats["Width"]["mean"]
                    }
                return self.stage_stats

        # Otherwise compute from geometry_case_data.json and CSVs
        print("Computing per-stage statistics...")

        # 1. Build case_id -> diagnosis from CSVs
        case_to_diagnosis = {}
        csv_files = {
            "train": os.path.join(self.base_dir, "train_1.csv"),
            "test": os.path.join(self.base_dir, "test.csv"),
            "val": os.path.join(self.base_dir, "valid.csv")
        }
        
        for split_name, csv_path in csv_files.items():
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    case_id = row['id_code']
                    diagnosis = int(row['diagnosis'])
                    case_to_diagnosis[case_id] = diagnosis
        
        # 2. Load geometry data
        if not os.path.exists(self.geometry_data_file):
            raise FileNotFoundError(
                f"Geometry data file not found: {self.geometry_data_file}\n"
                "Run geometry extraction script first."
            )

        with open(self.geometry_data_file, 'r') as f:
            geometry_data = json.load(f)

        # 3. Group by class
        class_data = defaultdict(list)
        
        for case_info in geometry_data:
            case_id = case_info['case_id']
            if case_id in case_to_diagnosis:
                diagnosis = case_to_diagnosis[case_id]
                class_data[diagnosis].append({
                    'VTI': case_info['VTI'],
                    'Width': case_info['Width']
                })
        
        # 4. Per-class means
        self.stage_stats = {}
        for stage in sorted(class_data.keys()):
            cases = class_data[stage]
            vti_values = [c['VTI'] for c in cases]
            width_values = [c['Width'] for c in cases]
            
            self.stage_stats[stage] = {
                "VTI": float(np.mean(vti_values)),
                "Width": float(np.mean(width_values)),
                "count": len(cases)
            }
        
        print(f"Computed stats for {len(self.stage_stats)} stages")
        for stage, stats in sorted(self.stage_stats.items()):
            print(f"  Stage {stage}: VTI={stats['VTI']:.6f}, Width={stats['Width']:.6f}, "
                  f"Count={stats['count']}")
        
        return self.stage_stats
    
    def recommend_target_geometry(
        self,
        source_stage: int,
        target_stage: int
    ) -> Dict[str, float]:
        """
        Recommend target VTI and width for source_stage -> target_stage.
        Args: source_stage, target_stage (0-4). Returns: {"target_vti", "target_width"}.
        """
        if self.stage_stats is None:
            self.load_stage_statistics()

        if target_stage not in self.stage_stats:
            raise ValueError(f"Target stage {target_stage} not in stats")
        
        target_stats = self.stage_stats[target_stage]
        
        return {
            "target_vti": target_stats["VTI"],
            "target_width": target_stats["Width"]
        }
    
    def generate_stage_mask(
        self,
        mask,
        source_stage: int,
        target_stage: int,
        **kwargs
    ) -> tuple:
        """
        Generate mask for target_stage.
        Args: mask, source_stage, target_stage; **kwargs passed to generate_curved_mask.
        Returns: (mask_array, metrics_dict).
        """
        if source_stage == target_stage:
            # Same stage: return original mask
            mask_array = self.mask_generator.load_mask(mask)
            baseline = self.mask_generator.get_baseline_geometry(mask)
            return mask_array, {
                "source_stage": source_stage,
                "target_stage": target_stage,
                "baseline_vti": baseline["VTI"],
                "baseline_width": baseline["Width"],
                "achieved_vti": baseline["VTI"],
                "achieved_width": baseline["Width"]
            }
        
        target_geometry = self.recommend_target_geometry(source_stage, target_stage)

        # Generate mask
        mask_out, metrics = self.mask_generator.generate_curved_mask(
            mask,
            target_geometry["target_vti"],
            target_geometry["target_width"],
            **kwargs
        )
        
        # Add stage info
        metrics["source_stage"] = source_stage
        metrics["target_stage"] = target_stage
        
        return mask_out, metrics
    
    def generate_all_stages(
        self,
        mask,
        source_stage: int,
        **kwargs
    ) -> Dict[int, tuple]:
        """
        Generate masks for all stages (0-4) from source_stage mask.
        Args: mask, source_stage; **kwargs to generate_curved_mask.
        Returns: {stage: (mask_array, metrics_dict)}.
        """
        results = {}

        if self.stage_stats is None:
            self.load_stage_statistics()

        for target_stage in sorted(self.stage_stats.keys()):
            print(f"\nGenerating mask for Stage {target_stage}...")
            mask_out, metrics = self.generate_stage_mask(
                mask, source_stage, target_stage, **kwargs
            )
            results[target_stage] = (mask_out, metrics)
            print(f"  Stage {target_stage}: VTI={metrics.get('achieved_vti', 'N/A'):.6f}, "
                  f"Width={metrics.get('achieved_width', 'N/A'):.6f}")
        
        return results
