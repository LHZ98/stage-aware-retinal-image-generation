"""
Vessel mask generator: geometry transform with precise VTI and width control.
Supports differentiated control (main branch vs capillaries) and exact VTI matching.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.vessel.skeleton_utils import (
    extract_skeleton_and_segments,
    get_branch_points,
    get_endpoints,
    compute_skeleton,
    compute_distance_transform
)
from main.vessel.tortuosity_control import (
    perturb_segments,
    _arc_length_parameterization,
    _tangent_at,
    _normal_2d,
    OMEGA
)
from main.vessel.width_control import (
    scale_segments_radii,
    reconstruct_mask_from_segments
)
from main.vessel.metrics import (
    compute_global_tortuosity,
    compute_global_width,
    segment_arc_length
)


class MaskGenerator:
    """
    Vessel mask generator with precise VTI and width control.
    """

    def __init__(self, min_segment_length: int = 5):
        """
        Initialize MaskGenerator.

        Args:
            min_segment_length: minimum segment length (pixels)
        """
        self.min_segment_length = min_segment_length
    
    def load_mask(self, mask_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load and binarize mask.

        Args:
            mask_input: mask path, numpy array, or PIL Image

        Returns:
            Binary mask (H, W), dtype=uint8, 0/1
        """
        if isinstance(mask_input, str):
            # File path
            ext = os.path.splitext(mask_input)[1].lower()
            if ext == ".npy":
                arr = np.load(mask_input)
            else:
                arr = np.array(Image.open(mask_input).convert("L"))
        elif isinstance(mask_input, Image.Image):
            arr = np.array(mask_input.convert("L"))
        else:
            arr = mask_input
        
        # Ensure 2D array
        if arr.ndim > 2:
            arr = arr[:, :, 0]

        # Binarize
        if arr.max() > 1:
            arr = (arr > 127).astype(np.uint8)
        else:
            arr = (arr > 0.5).astype(np.uint8)
        
        return arr
    
    def _smooth_mask_corners(self, mask: np.ndarray, sigma: float) -> np.ndarray:
        """Light Gaussian blur + threshold to smooth right angles/jaggies after reconstruction; keep binary."""
        if sigma <= 0:
            return mask
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return mask
        m = mask.astype(np.float64)
        m = gaussian_filter(m, sigma=sigma, mode="constant", cval=0)
        return (m > 0.5).astype(np.uint8)
    
    def classify_segments(
        self,
        segments: List[List[Tuple[Tuple[int, int], float]]],
        skeleton: np.ndarray,
        branch_points: np.ndarray,
        endpoints: np.ndarray
    ) -> Dict[str, List[int]]:
        """
        Classify segments into main branches and capillaries.

        Args:
            segments: list of segments
            skeleton: skeleton image
            branch_points: branch-point boolean array
            endpoints: endpoint boolean array

        Returns:
            {"main_branches": [indices], "capillaries": [indices]}
        """
        main_branches = []
        capillaries = []
        
        for i, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            
            start_point = seg[0][0]  # (r, c)
            end_point = seg[-1][0]   # (r, c)
            
            start_r, start_c = start_point
            end_r, end_c = end_point
            
            # Check endpoint type
            start_is_branch = (0 <= start_r < skeleton.shape[0] and 
                             0 <= start_c < skeleton.shape[1] and
                             branch_points[start_r, start_c])
            start_is_endpoint = (0 <= start_r < skeleton.shape[0] and 
                               0 <= start_c < skeleton.shape[1] and
                               endpoints[start_r, start_c])
            
            end_is_branch = (0 <= end_r < skeleton.shape[0] and 
                           0 <= end_c < skeleton.shape[1] and
                           branch_points[end_r, end_c])
            end_is_endpoint = (0 <= end_r < skeleton.shape[0] and 
                             0 <= end_c < skeleton.shape[1] and
                             endpoints[end_r, end_c])
            
            # Classification: main = both ends branch points; capillary = at least one endpoint or low connectivity
            if start_is_branch and end_is_branch:
                main_branches.append(i)
            elif start_is_endpoint or end_is_endpoint:
                capillaries.append(i)
            else:
                # Otherwise: use segment length; longer segments more likely main branch
                seg_length = segment_arc_length(seg)
                if seg_length > 50:  # tunable threshold
                    main_branches.append(i)
                else:
                    capillaries.append(i)
        
        return {
            "main_branches": main_branches,
            "capillaries": capillaries
        }
    
    def perturb_endpoint(
        self,
        endpoint: Tuple[int, int],
        segment: List[Tuple[Tuple[int, int], float]],
        is_start: bool,
        perturbation_range: Tuple[float, float] = (3.0, 5.0)
    ) -> Tuple[float, float]:
        """
        Perturb endpoint position within a small pixel range (e.g. 3-5 px).

        Args:
            endpoint: endpoint (r, c)
            segment: segment list
            is_start: whether this is the start endpoint
            perturbation_range: (min, max) perturbation in pixels

        Returns:
            New endpoint (r_new, c_new)
        """
        if len(segment) < 2:
            return endpoint
        
        # Tangent direction
        if is_start:
            # Use first two points
            p0 = segment[0][0]
            p1 = segment[1][0]
            dr = p1[0] - p0[0]
            dc = p1[1] - p0[1]
        else:
            # Use last two points
            p0 = segment[-2][0]
            p1 = segment[-1][0]
            dr = p1[0] - p0[0]
            dc = p1[1] - p0[1]
        
        # Normal (perpendicular to tangent)
        nr, nc = _normal_2d(dr, dc)
        
        if abs(nr) < 1e-9 and abs(nc) < 1e-9:
            return endpoint
        
        # Random perturbation magnitude and direction
        magnitude = np.random.uniform(perturbation_range[0], perturbation_range[1])
        direction = np.random.choice([-1, 1])

        # New position
        r_new = endpoint[0] + direction * magnitude * nr
        c_new = endpoint[1] + direction * magnitude * nc
        
        return (r_new, c_new)
    
    def apply_differentiated_tortuosity_control(
        self,
        segments: List[List[Tuple[Tuple[int, int], float]]],
        segment_types: Dict[str, List[int]],
        VTI_target: float,
        main_branch_vti_ratio: float = 0.3,
        capillary_vti_ratio: float = 1.7,
        preserve_branch_points: bool = True,
        endpoint_perturbation_range: Tuple[float, float] = (3.0, 5.0),
        skeleton: Optional[np.ndarray] = None,
        branch_points: Optional[np.ndarray] = None,
        tolerance: float = 1e-5,
        max_iter: int = 200
    ) -> Tuple[List[List[Tuple[Tuple[int, int], float]]], float]:
        """
        Apply differentiated VTI control (different strategy for main vs capillaries).
        Iterative optimization to reach target VTI.

        Args:
            segments: segment list
            segment_types: segment classification result
            VTI_target: target VTI
            main_branch_vti_ratio: main-branch amplitude ratio (relative to target)
            capillary_vti_ratio: capillary amplitude ratio (relative to target)
            preserve_branch_points: whether to keep branch points fixed
            endpoint_perturbation_range: endpoint perturbation range
            tolerance: VTI convergence tolerance
            max_iter: max iterations

        Returns:
            (perturbed_segments, achieved_VTI)
        """
        main_indices = segment_types["main_branches"]
        capillary_indices = segment_types["capillaries"]
        
        main_segments = [segments[i] for i in main_indices]
        capillary_segments = [segments[i] for i in capillary_indices]
        
        # Length weights
        L_main = sum(segment_arc_length(seg) for seg in main_segments) if main_segments else 0.0
        L_cap = sum(segment_arc_length(seg) for seg in capillary_segments) if capillary_segments else 0.0
        L_total = L_main + L_cap
        
        if L_total < 1e-9:
            return segments, compute_global_tortuosity(segments)
        
        # Same target VTI for main and capillaries; differentiated via amplitude ratios
        VTI_main_target = VTI_target
        VTI_cap_target = VTI_target

        A_main_scale = main_branch_vti_ratio  # e.g. 0.3 -> smaller main amplitude
        A_cap_scale = capillary_vti_ratio     # e.g. 1.7 -> larger capillary amplitude

        A_main = 0.0
        A_cap = 0.0
        learning_rate_main = 30.0
        learning_rate_cap = 30.0
        
        best_segments = segments
        best_VTI = compute_global_tortuosity(segments)
        best_error = abs(best_VTI - VTI_target)
        
        for iteration in range(max_iter):
            # Perturb main and capillary segments with amplitude scales
            if main_segments:
                main_perturbed = perturb_segments(main_segments, A_main * A_main_scale, length_weighted=True)
            else:
                main_perturbed = []
            
            if capillary_segments:
                cap_perturbed = perturb_segments(capillary_segments, A_cap * A_cap_scale, length_weighted=True)
            else:
                cap_perturbed = []
            
            # Apply endpoint perturbation (capillaries only; true endpoints only, not branch points)
            if preserve_branch_points and capillary_segments and branch_points is not None:
                cap_perturbed_with_endpoints = []
                for seg in cap_perturbed:
                    new_seg = list(seg)
                    # Check and perturb endpoint
                    if len(seg) >= 2:
                        start_point = seg[0][0]
                        end_point = seg[-1][0]
                        
                        start_r, start_c = int(round(start_point[0])), int(round(start_point[1]))
                        end_r, end_c = int(round(end_point[0])), int(round(end_point[1]))
                        
                        # Only perturb true endpoints (not branch points)
                        if (0 <= start_r < branch_points.shape[0] and 
                            0 <= start_c < branch_points.shape[1] and
                            not branch_points[start_r, start_c]):
                            # Perturb start endpoint
                            start_new = self.perturb_endpoint(
                                start_point, seg, is_start=True,
                                perturbation_range=endpoint_perturbation_range
                            )
                            new_seg[0] = (start_new, seg[0][1])
                        
                        if (0 <= end_r < branch_points.shape[0] and 
                            0 <= end_c < branch_points.shape[1] and
                            not branch_points[end_r, end_c]):
                            # Perturb end endpoint
                            end_new = self.perturb_endpoint(
                                end_point, seg, is_start=False,
                                perturbation_range=endpoint_perturbation_range
                            )
                            new_seg[-1] = (end_new, seg[-1][1])
                    
                    cap_perturbed_with_endpoints.append(new_seg)
                cap_perturbed = cap_perturbed_with_endpoints
            
            # Merge segments (preserve original order)
            all_perturbed = []
            main_idx = 0
            cap_idx = 0
            
            for i in range(len(segments)):
                if i in main_indices:
                    if main_idx < len(main_perturbed):
                        all_perturbed.append(main_perturbed[main_idx])
                        main_idx += 1
                    else:
                        all_perturbed.append(segments[i])
                elif i in capillary_indices:
                    if cap_idx < len(cap_perturbed):
                        all_perturbed.append(cap_perturbed[cap_idx])
                        cap_idx += 1
                    else:
                        all_perturbed.append(segments[i])
                else:
                    all_perturbed.append(segments[i])
            
            # Compute global VTI
            VTI_achieved = compute_global_tortuosity(all_perturbed)
            error = VTI_achieved - VTI_target
            
            # Update best result
            if abs(error) < best_error:
                best_error = abs(error)
                best_VTI = VTI_achieved
                best_segments = all_perturbed
            
            # Check convergence
            if abs(error) <= tolerance:
                return all_perturbed, VTI_achieved
            
            if abs(error) > 0.05:
                step_scale = 1.0
            elif abs(error) > 0.01:
                step_scale = 0.5
            else:
                step_scale = 0.1

            if error > 0:
                step_main = -learning_rate_main * step_scale
                step_cap = -learning_rate_cap * step_scale
            else:
                step_main = learning_rate_main * step_scale
                step_cap = learning_rate_cap * step_scale

            A_main += step_main
            A_cap += step_cap

            A_max_limit = 1000.0
            A_main = max(0.0, min(A_main, A_max_limit))
            A_cap = max(0.0, min(A_cap, A_max_limit))
            
            # Learning rate decay
            if iteration > 50:
                if abs(error) < 0.001:
                    learning_rate_main *= 0.95
                    learning_rate_cap *= 0.95
                elif abs(error) > 0.05:
                    # If error still large, slightly increase learning rate
                    learning_rate_main *= 1.01
                    learning_rate_cap *= 1.01
        
        return best_segments, best_VTI
    
    def apply_differentiated_width_control(
        self,
        segments: List[List[Tuple[Tuple[int, int], float]]],
        segment_types: Dict[str, List[int]],
        target_width: float,
        baseline_width: float,
        main_branch_width_ratio: float = 1.2,
        capillary_width_ratio: float = 0.8
    ) -> List[List[Tuple[Tuple[int, int], float]]]:
        """
        Apply differentiated width control (main vs capillaries).

        Args:
            segments: segment list
            segment_types: segment classification
            target_width: target width
            baseline_width: baseline width
            main_branch_width_ratio: main-branch width ratio
            capillary_width_ratio: capillary width ratio

        Returns:
            Width-scaled segments
        """
        main_indices = set(segment_types["main_branches"])
        capillary_indices = set(segment_types["capillaries"])

        alpha_base = target_width / baseline_width if baseline_width > 0 else 1.0
        alpha_base = max(0.95, min(1.05, alpha_base))
        alpha_main = alpha_base * main_branch_width_ratio
        alpha_cap = alpha_base * capillary_width_ratio
        alpha_main = max(0.9, min(1.1, alpha_main))
        alpha_cap = max(0.9, min(1.1, alpha_cap))
        
        scaled_segments = []
        for i, seg in enumerate(segments):
            if i in main_indices:
                alpha = alpha_main
            elif i in capillary_indices:
                alpha = alpha_cap
            else:
                alpha = alpha_base
            
            scaled_seg = [((p[0], p[1]), r * alpha) for p, r in seg]
            scaled_segments.append(scaled_seg)
        
        return scaled_segments
    
    def generate_curved_mask(
        self,
        mask: Union[str, np.ndarray, Image.Image],
        target_vti: float,
        target_width: float,
        preserve_branch_points: bool = True,
        endpoint_perturbation_range: Tuple[float, float] = (3.0, 5.0),
        main_branch_vti_ratio: float = 0.3,
        capillary_vti_ratio: float = 1.7,
        main_branch_width_ratio: float = 1.05,
        capillary_width_ratio: float = 0.95,
        verify_after_reconstruction: bool = True,
        tolerance: float = 0.02,
        random_selection_ratio: float = 1.0,
        random_omega: bool = True,
        random_amplitude: bool = True,
        smooth_after_reconstruction: bool = True,
        tortuosity_alpha: float = 1.9,
        tortuosity_beta: float = 0.85,
        random_seed: Optional[int] = None,
        use_vessel_style: bool = False,
        smooth_centerline_window: int = 5,
        smooth_centerline_iterations: int = 1,
        amplitude_scale_range: Tuple[float, float] = (0.6, 1.4),
        smooth_mask_corners_sigma: float = 1.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a more curved vessel mask with precise VTI and width control.

        Args:
            mask: input mask (path, numpy array, or PIL Image)
            target_vti: target VTI
            target_width: target width (pixels)
            preserve_branch_points: whether to keep branch points fixed
            endpoint_perturbation_range: endpoint perturbation range
            main_branch_vti_ratio, capillary_vti_ratio: VTI amplitude ratios
            main_branch_width_ratio, capillary_width_ratio: width ratios
            verify_after_reconstruction: verify VTI after reconstruction
            tolerance: VTI tolerance (default 0.02)
            random_selection_ratio: fraction of segments to perturb (0-1)
            random_omega: per-segment random frequency
            random_amplitude: per-segment random amplitude scale
            smooth_after_reconstruction: smooth mask after reconstruction
            tortuosity_alpha, tortuosity_beta: endpoint/main amplitude weights
            random_seed: fixed seed for reproducibility
            use_vessel_style: use path sine + centerline smooth + reconstruct
            smooth_centerline_window, smooth_centerline_iterations: for vessel_style
            amplitude_scale_range: per-segment amplitude range for vessel_style
            smooth_mask_corners_sigma: Gaussian sigma to smooth corners; 0 = off

        Returns:
            (generated_mask, metrics_dict)
        """
        # 1. Load and validate mask
        mask_array = self.load_mask(mask)
        if mask_array.sum() == 0:
            raise ValueError("Input mask is empty (no vessel pixels)")
        
        shape = mask_array.shape
        
        # 2. Extract skeleton and segments
        skeleton, dist, segments, _ = extract_skeleton_and_segments(
            mask_array, min_length=self.min_segment_length
        )
        
        if len(segments) == 0:
            raise ValueError("Could not extract valid segments")
        
        # 3. Get branch points and endpoints
        branch_points = get_branch_points(skeleton)
        endpoints = get_endpoints(skeleton)
        
        # 4. Classify segments (main vs capillaries)
        segment_classification = self.classify_segments(
            segments, skeleton, branch_points, endpoints
        )
        # Tortuosity: capillaries use endpoint (larger amp, lower freq), main use main
        n_seg = len(segments)
        segment_types_list = ["main"] * n_seg
        for i in segment_classification.get("capillaries", []):
            if i < n_seg:
                segment_types_list[i] = "endpoint"
        
        # 5. Baseline geometry
        baseline_vti = compute_global_tortuosity(segments)
        baseline_width = compute_global_width(segments)
        
        # 6. Tortuosity control
        from main.vessel.tortuosity_control import (
            apply_tortuosity_control,
            apply_path_sine_to_match_vti,
            smooth_segments_centerlines,
        )
        if use_vessel_style:
            # Fix branch-point ends only; leave skeleton endpoint (leaf) ends free to perturb
            H, W = skeleton.shape
            segment_fix_ends = []
            for seg in segments:
                (r0, c0), _ = seg[0]
                (r1, c1), _ = seg[-1]
                ir0, ic0 = int(round(r0)), int(round(c0))
                ir1, ic1 = int(round(r1)), int(round(c1))
                fix_first = bool(branch_points[ir0, ic0]) if (0 <= ir0 < H and 0 <= ic0 < W) else True
                fix_last = bool(branch_points[ir1, ic1]) if (0 <= ir1 < H and 0 <= ic1 < W) else True
                segment_fix_ends.append((fix_first, fix_last))
            # Vessel style: path sine with random phase and amplitude in range
            try:
                seg_tort, achieved_vti_seg, _ = apply_path_sine_to_match_vti(
                    segments,
                    segment_types_list,
                    VTI_target=target_vti,
                    random_seed=random_seed,
                    alpha_endpoint=tortuosity_alpha,
                    beta_main=tortuosity_beta,
                    amplitude_random_scale=True,
                    amplitude_scale_range=amplitude_scale_range,
                    phase_jitter_scale=0.6,
                    segment_fix_ends=segment_fix_ends,
                )
                # After perturb: smooth centerline then restore (vessel sine_smooth_centerline)
                seg_tort = smooth_segments_centerlines(
                    seg_tort,
                    window=smooth_centerline_window,
                    iterations=smooth_centerline_iterations,
                )
            except Exception:
                seg_tort, achieved_vti_seg = apply_tortuosity_control(
                    segments,
                    VTI_target=target_vti,
                    segment_types=segment_types_list,
                    alpha=tortuosity_alpha,
                    beta=tortuosity_beta,
                    random_omega=random_omega,
                    random_amplitude=random_amplitude,
                    random_seed=random_seed,
                )
        else:
            seg_tort, achieved_vti_seg = apply_tortuosity_control(
                segments,
                VTI_target=target_vti,
                segment_types=segment_types_list,
                alpha=tortuosity_alpha,
                beta=tortuosity_beta,
                random_omega=random_omega,
                random_amplitude=random_amplitude,
                random_seed=random_seed,
            )
        
        # Step 2: Width control - scale all segment radii
        from main.vessel.width_control import scale_segments_radii
        alpha = target_width / baseline_width if baseline_width > 0 else 1.0
        seg_final = scale_segments_radii(seg_tort, alpha)
        
        # Step 3: Reconstruct mask
        mask_out = reconstruct_mask_from_segments(seg_final, shape)
        # Optional: smooth corners at segment junctions (light blur + threshold)
        if use_vessel_style and smooth_mask_corners_sigma > 0:
            mask_out = self._smooth_mask_corners(mask_out, smooth_mask_corners_sigma)
        
        achieved_vti = float(achieved_vti_seg)  # VTI from perturbed centerline
        width_final = compute_global_width(seg_final)

        # 10. Return mask and metrics
        metrics = {
            "baseline_vti": float(baseline_vti),
            "baseline_width": float(baseline_width),
            "target_vti": float(target_vti),
            "achieved_vti": float(achieved_vti),
            "target_width": float(target_width),
            "achieved_width": float(width_final),
            "vti_error": float(abs(achieved_vti - target_vti)),
            "width_error": float(abs(width_final - target_width)),
            "alpha": float(alpha)
        }
        
        return mask_out, metrics
    
    def get_baseline_geometry(self, mask: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Get baseline geometry (VTI, width, density) for a mask.

        Args:
            mask: input mask

        Returns:
            dict with VTI, Width, Density
        """
        mask_array = self.load_mask(mask)
        skeleton, dist, segments, _ = extract_skeleton_and_segments(
            mask_array, min_length=self.min_segment_length
        )
        
        vti = compute_global_tortuosity(segments)
        width = compute_global_width(segments)
        density = skeleton.sum() / mask_array.size
        
        return {
            "VTI": float(vti),
            "Width": float(width),
            "Density": float(density)
        }
    
    def generate_from_geometry(
        self,
        mask: Union[str, np.ndarray, Image.Image],
        target_geometry: List[float]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate mask from geometry vector (convenience wrapper).
        
        Args:
            mask: input mask
            target_geometry: [VTI, width, density] (density ignored)

        Returns:
            (generated_mask, metrics_dict)
        """
        target_vti = target_geometry[0]
        target_width = target_geometry[1]
        
        return self.generate_curved_mask(mask, target_vti, target_width)
