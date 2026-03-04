"""
Stage 6 & 7: Independence analysis and experiment grid.

- Stage 6: Modify tortuosity only -> measure width change; modify width only -> measure tortuosity change.
- Stage 7: Grid over VTI_target in [baseline, baseline*1.05, baseline*1.10] and
  width scale in [0.8, 1.0, 1.2]. Save masks, metrics, visualizations, summary table.

Run from repo root: python -m main.vessel.experiment --mask path/to/mask.npy --out_dir path/to/output
Or: python -m main.vessel.experiment --mask path/to/mask.png --out_dir path/to/output
(Mask image: will be binarized with threshold > 0.5)
"""

import argparse
import json
import os
from typing import List, Tuple, Any, Optional

import numpy as np

from .skeleton_utils import extract_skeleton_and_segments
from .metrics import compute_global_tortuosity, compute_global_width, compute_global_caliber
from .tortuosity_control import (
    apply_tortuosity_control,
    perturb_segments_along_paths,
    apply_path_sine_to_match_vti,
    smooth_segments_centerlines,
)
from .width_control import (
    scale_segments_radii,
    reconstruct_mask_from_segments,
    apply_width_control,
    enforce_murray_at_branches,
)
from .fractal_dimension import fractal_dimension_box_counting
from .neovascularization import add_small_vessels_at_endpoints


def load_mask(path: str) -> np.ndarray:
    """Load binary mask from .npy or image; return (H, W) with 0/1."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        try:
            from PIL import Image
            arr = np.array(Image.open(path))
        except Exception:
            import imageio
            arr = imageio.imread(path)
        if arr.ndim > 2:
            arr = arr[:, :, 0]
        arr = (arr > 0.5).astype(np.uint8)
    return arr


def run_independence_analysis(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    segment_types: List[str],
    shape: Tuple[int, int],
) -> dict:
    """
    Stage 6: Tortuosity-only and caliber-only changes; report cross-factor error.
    """
    T0 = compute_global_tortuosity(segments)
    C0 = compute_global_caliber(segments)

    # (1) Tortuosity only: target 1.05 * T0, no caliber scaling
    seg_tort, T_achieved = apply_tortuosity_control(
        segments, VTI_target=1.05 * T0, segment_types=segment_types
    )
    C_after_tort = compute_global_caliber(seg_tort)
    caliber_change_tort_only = abs(C_after_tort - C0)

    # (2) Caliber only: lambda=1.2, no tortuosity change; re-extract from new mask
    mask_caliber_only = apply_width_control(segments, alpha=1.2, shape=shape)
    _, _, seg_from_mask, _ = extract_skeleton_and_segments(mask_caliber_only)
    T_after_caliber = compute_global_tortuosity(seg_from_mask) if seg_from_mask else T0
    tort_change_caliber_only = abs(T_after_caliber - T0)

    return {
        "baseline_T": float(T0),
        "baseline_C": float(C0),
        "tort_only_target_T": float(1.05 * T0),
        "tort_only_achieved_T": float(T_achieved),
        "tort_only_C_after": float(C_after_tort),
        "cross_factor_error_caliber_when_tort_changed": float(caliber_change_tort_only),
        "caliber_only_lambda": 1.2,
        "caliber_only_T_after": float(T_after_caliber),
        "cross_factor_error_tort_when_caliber_changed": float(tort_change_caliber_only),
    }


def run_single_experiment(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    segment_types: List[str],
    shape: Tuple[int, int],
    VTI_target: float,
    alpha: float,
    add_neovascularization: bool = False,
    neovascularization_prob: float = 0.1,
    neovascularization_max_length: Optional[float] = None,
    random_seed: Optional[int] = None,
    use_fixed_amplitude: bool = False,
    fixed_amplitude: Optional[float] = None,
    use_raw_amplitude: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    One (VTI_target, lambda): tortuosity control [-> optional neovascularization]
    -> scale radii by lambda -> Murray at branches -> reconstruct mask.
    use_raw_amplitude: if True with use_fixed_amplitude, A = fixed_amplitude directly (no VTI scaling).
    """
    seg_tort, T_achieved_tort = apply_tortuosity_control(
        segments,
        VTI_target=VTI_target,
        segment_types=segment_types,
        random_seed=random_seed,
        use_fixed_amplitude=use_fixed_amplitude,
        fixed_amplitude=fixed_amplitude,
        use_raw_amplitude=use_raw_amplitude,
    )
    if add_neovascularization:
        seg_tort = add_small_vessels_at_endpoints(
            seg_tort, shape,
            prob_segment=neovascularization_prob,
            prob_at_endpoint=0.35,
            max_new_vessel_length=neovascularization_max_length,
            phase_jitter_scale=0.6,
            random_seed=random_seed,
        )
    seg_scaled = scale_segments_radii(seg_tort, alpha)
    seg_final = enforce_murray_at_branches(seg_scaled)
    mask_out = reconstruct_mask_from_segments(seg_final, shape)
    achieved_T = float(T_achieved_tort)
    achieved_C = compute_global_caliber(seg_final)
    skel_out, _, _, _ = extract_skeleton_and_segments(mask_out)
    fractal_d = fractal_dimension_box_counting(skel_out)
    C_baseline = compute_global_caliber(segments)
    target_C = alpha * C_baseline
    return mask_out, {
        "target_VTI": float(VTI_target),
        "achieved_VTI": achieved_T,
        "target_caliber": float(target_C),
        "achieved_caliber": float(achieved_C),
        "fractal_dimension": float(fractal_d),
        "lambda": float(alpha),
    }


def save_visualization(
    original: np.ndarray,
    transformed: np.ndarray,
    path: str,
) -> None:
    """Save side-by-side or overlay comparison (e.g. 2-panel image)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(transformed, cmap="gray")
    axes[1].set_title("Transformed")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_experiment_grid(
    mask: np.ndarray,
    out_dir: str,
    vti_multipliers: Optional[List[float]] = None,
    width_scales: Optional[List[float]] = None,
    add_neovascularization: bool = False,
    neovascularization_prob: float = 0.1,
    neovascularization_max_length: Optional[float] = None,
    base_random_seed: Optional[int] = 42,
    use_fixed_amplitude: bool = False,
    fixed_amplitude: Optional[float] = None,
) -> None:
    """
    Stage 7: Full grid, save masks, metrics, visualizations, summary table.

    vti_multipliers: multipliers of baseline; width_scales: caliber scale lambda.
    add_neovascularization: add small neo at terminals; neovascularization_prob: prob per terminal.
    use_fixed_amplitude: if True use fixed A (no VTI bisection); fixed_amplitude None -> 0.6*min_radius.
    base_random_seed: for tortuosity/neo; sub-seeds per (VTI, alpha) in grid.
    """
    os.makedirs(out_dir, exist_ok=True)
    shape = mask.shape
    _, _, segments, segment_types = extract_skeleton_and_segments(mask)
    if not segments:
        raise RuntimeError("No segments extracted from mask (too few or too short?).")

    T_baseline = compute_global_tortuosity(segments)
    C_baseline = compute_global_caliber(segments)

    if vti_multipliers is None:
        vti_multipliers = [1.0, 1.05, 1.10]
    if width_scales is None:
        width_scales = [0.8, 1.0, 1.2]

    # Stage 6
    indep = run_independence_analysis(segments, segment_types, shape)
    with open(os.path.join(out_dir, "independence_analysis.json"), "w") as f:
        json.dump(indep, f, indent=2)

    # Stage 7 grid
    VTI_targets = [T_baseline * m for m in vti_multipliers]
    rows = []
    run_idx = 0
    for VTI_target in VTI_targets:
        for alpha in width_scales:
            seed = (base_random_seed + run_idx) if base_random_seed is not None else None
            run_idx += 1
            mask_out, metrics = run_single_experiment(
                segments, segment_types, shape, VTI_target, alpha,
                add_neovascularization=add_neovascularization,
                neovascularization_prob=neovascularization_prob,
                neovascularization_max_length=neovascularization_max_length,
                random_seed=seed,
                use_fixed_amplitude=use_fixed_amplitude,
                fixed_amplitude=fixed_amplitude,
            )
            vid = f"VTI{VTI_target:.4f}_alpha{alpha:.2f}".replace(".", "_")
            np.save(os.path.join(out_dir, f"mask_{vid}.npy"), mask_out)
            with open(os.path.join(out_dir, f"metrics_{vid}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            save_visualization(
                mask,
                mask_out,
                os.path.join(out_dir, f"viz_{vid}.png"),
            )
            rows.append({
                "Target VTI": metrics["target_VTI"],
                "Achieved VTI": metrics["achieved_VTI"],
                "Target Caliber": metrics["target_caliber"],
                "Achieved Caliber": metrics["achieved_caliber"],
                "Fractal Dimension": metrics["fractal_dimension"],
            })

    # Summary table (CSV)
    import csv
    summary_path = os.path.join(out_dir, "summary_table.csv")
    fieldnames = ["Target VTI", "Achieved VTI", "Target Caliber", "Achieved Caliber", "Fractal Dimension"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Also save as readable text
    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.write("Target VTI | Achieved VTI | Target Caliber | Achieved Caliber | Fractal Dimension\n")
        f.write("-" * 80 + "\n")
        for r in rows:
            f.write(f"{r['Target VTI']:.4f} | {r['Achieved VTI']:.4f} | {r['Target Caliber']:.4f} | {r['Achieved Caliber']:.4f} | {r['Fractal Dimension']:.4f}\n")
    return


def run_sine_path_vti_grid(
    mask: np.ndarray,
    out_dir: str,
    vti_multipliers: Optional[List[float]] = None,
    width_scales: Optional[List[float]] = None,
    add_neovascularization: bool = False,
    neovascularization_prob: float = 0.1,
    neovascularization_max_length: Optional[float] = None,
    base_random_seed: Optional[int] = 42,
    sine_k_cycles: float = 1.0,
    sine_alpha_endpoint: float = 0.9,
    sine_beta_main: float = 1.0,
    sine_main_length_gain: float = 0.0,
    sine_center_protect_radius: float = 0.0,
    sine_center_protect_strength: float = 0.0,
    sine_smooth_centerline: bool = False,
    sine_smooth_window: int = 5,
    sine_smooth_iterations: int = 1,
) -> None:
    """
    Path sine + VTI loop: for each VTI_target bisect amplitude A so achieved T approx target; then capsule reconstruct, optional neo. vti_multipliers = multipliers of baseline.
    """
    import csv
    os.makedirs(out_dir, exist_ok=True)
    shape = mask.shape
    _, _, segments, segment_types = extract_skeleton_and_segments(mask)
    if not segments:
        raise RuntimeError("No segments extracted from mask.")
    T_baseline = compute_global_tortuosity(segments)
    C_baseline = compute_global_caliber(segments)
    if vti_multipliers is None:
        vti_multipliers = [1.0, 1.05, 1.10]
    if width_scales is None:
        width_scales = [0.8, 1.0, 1.2]
    VTI_targets = [T_baseline * m for m in vti_multipliers]
    rows = []
    run_idx = 0
    for VTI_target in VTI_targets:
        for alpha in width_scales:
            seed = (base_random_seed + run_idx) if base_random_seed is not None else None
            run_idx += 1
            seg_pert, T_achieved, A_used = apply_path_sine_to_match_vti(
                segments,
                segment_types,
                VTI_target,
                random_seed=seed,
                K_cycles=sine_k_cycles,
                alpha_endpoint=sine_alpha_endpoint,
                beta_main=sine_beta_main,
                main_length_gain=sine_main_length_gain,
                center_protect_radius=sine_center_protect_radius,
                center_protect_strength=sine_center_protect_strength,
            )
            if add_neovascularization:
                seg_pert = add_small_vessels_at_endpoints(
                    seg_pert, shape,
                    prob_segment=neovascularization_prob,
                    prob_at_endpoint=0.35,
                    max_new_vessel_length=neovascularization_max_length,
                    phase_jitter_scale=0.6,
                    random_seed=seed,
                )
            if sine_smooth_centerline:
                seg_pert = smooth_segments_centerlines(
                    seg_pert,
                    window=sine_smooth_window,
                    iterations=sine_smooth_iterations,
                )
            seg_scaled = scale_segments_radii(seg_pert, alpha)
            seg_final = enforce_murray_at_branches(seg_scaled)
            mask_out = reconstruct_mask_from_segments(seg_final, shape)
            _, _, seg_out, _ = extract_skeleton_and_segments(mask_out)
            achieved_C = float(compute_global_caliber(seg_out)) if seg_out else 0.0
            skel_out, _, _, _ = extract_skeleton_and_segments(mask_out)
            fractal_d = float(fractal_dimension_box_counting(skel_out))
            target_C = alpha * C_baseline
            vid = f"pathVTI{VTI_target:.4f}_alpha{alpha:.2f}".replace(".", "_")
            if add_neovascularization:
                vid = vid + "_neo"
            np.save(os.path.join(out_dir, f"mask_{vid}.npy"), mask_out)
            metrics = {
                "target_VTI": float(VTI_target),
                "achieved_VTI": float(T_achieved),
                "target_caliber": float(target_C),
                "achieved_caliber": achieved_C,
                "fractal_dimension": fractal_d,
                "lambda": float(alpha),
                "amplitude_used": float(A_used),
            }
            with open(os.path.join(out_dir, f"metrics_{vid}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            save_visualization(mask, mask_out, os.path.join(out_dir, f"viz_{vid}.png"))
            rows.append({
                "Target VTI": VTI_target,
                "Achieved VTI": T_achieved,
                "Target Caliber": target_C,
                "Achieved Caliber": achieved_C,
                "Fractal Dimension": fractal_d,
                "Amplitude": A_used,
            })
    summary_path = os.path.join(out_dir, "summary_table.csv")
    fieldnames = ["Target VTI", "Achieved VTI", "Target Caliber", "Achieved Caliber", "Fractal Dimension", "Amplitude"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.write("Target VTI | Achieved VTI | Target Caliber | Achieved Caliber | Fractal Dim | Amplitude\n")
        f.write("-" * 85 + "\n")
        for r in rows:
            f.write(f"{r['Target VTI']:.4f} | {r['Achieved VTI']:.4f} | {r['Target Caliber']:.4f} | {r['Achieved Caliber']:.4f} | {r['Fractal Dimension']:.4f} | {r['Amplitude']:.2f}\n")


def run_sine_only_visible(
    mask: np.ndarray,
    out_dir: str,
    amplitudes: Optional[List[float]] = None,
    random_seed: Optional[int] = 42,
    use_along_paths: bool = False,
    add_neovascularization: bool = False,
    neovascularization_prob: float = 0.1,
    neovascularization_max_length: Optional[float] = None,
    width_scale: float = 1.0,
    sine_k_cycles: float = 1.0,
    sine_alpha_endpoint: float = 0.9,
    sine_beta_main: float = 1.0,
    sine_main_length_gain: float = 0.0,
    sine_center_protect_radius: float = 0.0,
    sine_center_protect_strength: float = 0.0,
    sine_smooth_centerline: bool = False,
    sine_smooth_window: int = 5,
    sine_smooth_iterations: int = 1,
) -> None:
    """
    Sine perturbation only (no VTI target); fixed amplitude, visible, capsule reconstruction. use_along_paths: path-continuous sine + non-periodic phase. add_neovascularization: add neo with neovascularization_prob.
    """
    if amplitudes is None:
        amplitudes = [6.0, 10.0, 14.0]
    os.makedirs(out_dir, exist_ok=True)
    shape = mask.shape
    _, _, segments, segment_types = extract_skeleton_and_segments(mask)
    if not segments:
        raise RuntimeError("No segments extracted from mask.")
    import csv
    rows = []
    for i, A in enumerate(amplitudes):
        seed = (random_seed + i) if random_seed is not None else None
        if use_along_paths:
            seg_pert = perturb_segments_along_paths(
                segments,
                A=A,
                K_cycles=sine_k_cycles,
                phase_jitter_scale=0.6,
                random_seed=seed,
                segment_types=segment_types,
                alpha_endpoint=sine_alpha_endpoint,
                beta_main=sine_beta_main,
                main_length_gain=sine_main_length_gain,
                center_protect_radius=sine_center_protect_radius,
                center_protect_strength=sine_center_protect_strength,
            )
            if add_neovascularization:
                seg_pert = add_small_vessels_at_endpoints(
                    seg_pert, shape,
                    prob_segment=neovascularization_prob,
                    prob_at_endpoint=0.35,
                    max_new_vessel_length=neovascularization_max_length,
                    phase_jitter_scale=0.6,
                    random_seed=seed,
                )
            if sine_smooth_centerline:
                seg_pert = smooth_segments_centerlines(
                    seg_pert,
                    window=sine_smooth_window,
                    iterations=sine_smooth_iterations,
                )
            seg_scaled = scale_segments_radii(seg_pert, width_scale)
            seg_final = enforce_murray_at_branches(seg_scaled)
            mask_out = reconstruct_mask_from_segments(seg_final, shape)
            _, _, seg_out, _ = extract_skeleton_and_segments(mask_out)
            achieved_T = float(compute_global_tortuosity(seg_out)) if seg_out else 1.0
            achieved_C = float(compute_global_caliber(seg_out)) if seg_out else 0.0
            skel_out, _, _, _ = extract_skeleton_and_segments(mask_out)
            fractal_d = float(fractal_dimension_box_counting(skel_out))
            metrics = {
                "target_VTI": 1.0,
                "achieved_VTI": achieved_T,
                "target_caliber": float(compute_global_caliber(segments)),
                "achieved_caliber": achieved_C,
                "fractal_dimension": fractal_d,
                "lambda": 1.0,
            }
        else:
            mask_out, metrics = run_single_experiment(
                segments,
                segment_types,
                shape,
                VTI_target=1.0,
                alpha=1.0,
                random_seed=seed,
                use_fixed_amplitude=True,
                fixed_amplitude=A,
                use_raw_amplitude=True,
            )
        vid = f"sinA{A:.1f}".replace(".", "_")
        if use_along_paths:
            vid = "path_" + vid
        if use_along_paths and add_neovascularization:
            vid = vid + "_neo"
        np.save(os.path.join(out_dir, f"mask_{vid}.npy"), mask_out)
        with open(os.path.join(out_dir, f"metrics_{vid}.json"), "w") as f:
            json.dump({**metrics, "amplitude": float(A)}, f, indent=2)
        save_visualization(mask, mask_out, os.path.join(out_dir, f"viz_{vid}.png"))
        rows.append({
            "Amplitude": A,
            "Achieved VTI": metrics["achieved_VTI"],
            "Achieved Caliber": metrics["achieved_caliber"],
            "Fractal Dimension": metrics["fractal_dimension"],
        })
    with open(os.path.join(out_dir, "summary_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Amplitude", "Achieved VTI", "Achieved Caliber", "Fractal Dimension"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.write("Amplitude | Achieved VTI | Achieved Caliber | Fractal Dimension\n")
        f.write("-" * 60 + "\n")
        for r in rows:
            f.write(f"{r['Amplitude']:.1f} | {r['Achieved VTI']:.4f} | {r['Achieved Caliber']:.4f} | {r['Fractal Dimension']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Vessel mask transformation experiments")
    parser.add_argument("--mask", type=str, required=True, help="Path to binary vessel mask (.npy or image)")
    parser.add_argument("--out_dir", type=str, default="vessel_experiment_output", help="Output directory")
    parser.add_argument(
        "--vti_targets",
        type=float,
        nargs="+",
        default=None,
        help="VTI target = baseline × multiplier, e.g. 1.0 1.05 1.10 (default)",
    )
    parser.add_argument(
        "--width_scales",
        type=float,
        nargs="+",
        default=None,
        help="Width scale alpha, e.g. 0.8 1.0 1.2 (default)",
    )
    parser.add_argument(
        "--add_neovascularization",
        action="store_true",
        help="Add small neovascularization at terminals (optional)",
    )
    parser.add_argument(
        "--neovascularization_prob",
        type=float,
        default=0.1,
        help="Per-segment prob to add neo (default 0.1); then 0.35 at endpoint, 0.65 mid-segment",
    )
    parser.add_argument(
        "--neovascularization_max_length",
        type=float,
        default=None,
        help="Max neo length in pixels; omit for default range",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for tortuosity and neo (default 42)",
    )
    parser.add_argument(
        "--use_fixed_amplitude",
        action="store_true",
        help="Fixed amplitude: no VTI bisection; A from --fixed_amplitude or 0.6*min_radius",
    )
    parser.add_argument(
        "--fixed_amplitude",
        type=float,
        default=None,
        metavar="A",
        help="Fixed amplitude A (px); default 0.6*min_radius in fixed-amplitude mode",
    )
    parser.add_argument(
        "--sine_only_visible",
        action="store_true",
        help="Sine perturbation only (no VTI target); fixed amplitude, visible, capsule reconstruction",
    )
    parser.add_argument(
        "--sine_only_amplitudes",
        type=float,
        nargs="+",
        default=None,
        metavar="A",
        help="Amplitude list for sine_only_visible (px), default 6 10 14",
    )
    parser.add_argument(
        "--sine_along_paths",
        action="store_true",
        help="With sine_only_visible: path-continuous sine + non-periodic phase",
    )
    parser.add_argument(
        "--sine_path_vti",
        action="store_true",
        help="Path sine + VTI loop: bisect amplitude so achieved VTI ≈ target; use with --vti_targets",
    )
    parser.add_argument(
        "--sine_k_cycles",
        type=float,
        default=1.0,
        help="Path sine frequency (waves per path); lower = sparser, default 1.0",
    )
    parser.add_argument(
        "--sine_alpha_endpoint",
        type=float,
        default=0.9,
        help="Path sine endpoint amplitude factor, default 0.9",
    )
    parser.add_argument(
        "--sine_beta_main",
        type=float,
        default=1.0,
        help="Path sine main-branch amplitude factor, default 1.0",
    )
    parser.add_argument(
        "--sine_main_length_gain",
        type=float,
        default=0.0,
        help="Main length gain: A_main *= (1 + gain * L_seg/L_path), default 0.0",
    )
    parser.add_argument(
        "--sine_center_protect_radius",
        type=float,
        default=0.0,
        help="Center protection radius (px). >0 to enable; center = vessel density peak.",
    )
    parser.add_argument(
        "--sine_center_protect_strength",
        type=float,
        default=0.0,
        help="Center protection strength [0,1], default 0.0.",
    )
    parser.add_argument(
        "--sine_smooth_centerline",
        action="store_true",
        help="Smooth centerline before reconstruction (anchors fixed).",
    )
    parser.add_argument(
        "--sine_smooth_window",
        type=int,
        default=5,
        help="Centerline smooth window size (odd, default 5).",
    )
    parser.add_argument(
        "--sine_smooth_iterations",
        type=int,
        default=1,
        help="Centerline smooth iterations (default 1).",
    )
    parser.add_argument(
        "--sine_width_scale",
        type=float,
        default=1.0,
        help="Caliber scale when sine_only_visible (>1 slightly thicker), default 1.0.",
    )
    args = parser.parse_args()
    mask = load_mask(args.mask)
    if args.sine_path_vti:
        run_sine_path_vti_grid(
            mask,
            args.out_dir,
            vti_multipliers=args.vti_targets,
            width_scales=args.width_scales,
            add_neovascularization=args.add_neovascularization,
            neovascularization_prob=args.neovascularization_prob,
            neovascularization_max_length=args.neovascularization_max_length,
            base_random_seed=args.random_seed,
            sine_k_cycles=args.sine_k_cycles,
            sine_alpha_endpoint=args.sine_alpha_endpoint,
            sine_beta_main=args.sine_beta_main,
            sine_main_length_gain=args.sine_main_length_gain,
            sine_center_protect_radius=args.sine_center_protect_radius,
            sine_center_protect_strength=args.sine_center_protect_strength,
            sine_smooth_centerline=args.sine_smooth_centerline,
            sine_smooth_window=args.sine_smooth_window,
            sine_smooth_iterations=args.sine_smooth_iterations,
        )
        print("Done. Output in", args.out_dir)
        return
    if args.sine_only_visible:
        run_sine_only_visible(
            mask,
            args.out_dir,
            amplitudes=args.sine_only_amplitudes,
            random_seed=args.random_seed,
            use_along_paths=args.sine_along_paths,
            add_neovascularization=args.add_neovascularization,
            neovascularization_prob=args.neovascularization_prob,
            neovascularization_max_length=args.neovascularization_max_length,
            width_scale=args.sine_width_scale,
            sine_k_cycles=args.sine_k_cycles,
            sine_alpha_endpoint=args.sine_alpha_endpoint,
            sine_beta_main=args.sine_beta_main,
            sine_main_length_gain=args.sine_main_length_gain,
            sine_center_protect_radius=args.sine_center_protect_radius,
            sine_center_protect_strength=args.sine_center_protect_strength,
            sine_smooth_centerline=args.sine_smooth_centerline,
            sine_smooth_window=args.sine_smooth_window,
            sine_smooth_iterations=args.sine_smooth_iterations,
        )
        print("Done. Output in", args.out_dir)
        return
    use_fixed = args.use_fixed_amplitude or (args.fixed_amplitude is not None)
    run_experiment_grid(
        mask,
        args.out_dir,
        vti_multipliers=args.vti_targets,
        width_scales=args.width_scales,
        add_neovascularization=args.add_neovascularization,
        neovascularization_prob=args.neovascularization_prob,
        neovascularization_max_length=args.neovascularization_max_length,
        base_random_seed=args.random_seed,
        use_fixed_amplitude=use_fixed,
        fixed_amplitude=args.fixed_amplitude,
    )
    print("Done. Output in", args.out_dir)


if __name__ == "__main__":
    main()
