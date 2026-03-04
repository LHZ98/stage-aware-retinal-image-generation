"""
Stage 3: Tortuosity control via sinusoidal normal perturbation.

Sine perturbation:
  - Arc-length s in [0,1] per segment; tangent T and unit normal n (T rotated 90 deg CCW) at each point.
  - Only interior points perturbed; endpoints and branch points fixed:
      p_new = p + A * sin(ω*s) * n
  - omega: waves per segment (e.g. 6pi ~ 3 cycles), A: amplitude (px). Long segments use A_seg = A*(L_seg/L_avg) for target VTI.
  - Bisect A so perturbed centerline T_global approx VTI_target. Achieved VTI from this centerline, not re-skeletonized.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

from . import metrics
from . import path_utils
try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

# Angular frequency: lower -> fewer waves per segment, larger bend, more visible tortuosity
OMEGA = 6.0 * np.pi  # default fallback
OMEGA_ENDPOINT = 0.35 * np.pi   # ~0.18 cycle per segment
OMEGA_MAIN = 0.7 * np.pi       # ~0.35 cycle for main
# Random omega: endpoint low-freq large bend, main slightly higher
OMEGA_ENDPOINT_LO, OMEGA_ENDPOINT_HI = 0.2 * np.pi, 0.45 * np.pi
OMEGA_MAIN_LO, OMEGA_MAIN_HI = 0.45 * np.pi, 0.95 * np.pi
TOLERANCE = 5e-4
A_MAX = 800.0
MAX_ITER = 100
# Regional weights: endpoint larger amplitude, main also visible curvature
ALPHA_ENDPOINT = 1.2
BETA_MAIN = 0.7
# Amplitude randomness: scale in [0.7, 1.4] for natural variation
AMPLITUDE_SCALE_LO, AMPLITUDE_SCALE_HI = 0.7, 1.4


def _arc_length_parameterization(
    segment: List[Tuple[Tuple[int, int], float]],
) -> np.ndarray:
    """Return s[i] in [0, 1] for each point (cumulative arc / total arc)."""
    n = len(segment)
    if n < 2:
        return np.zeros(n)
    cum = np.zeros(n)
    for i in range(1, n):
        (r0, c0), _ = segment[i - 1]
        (r1, c1), _ = segment[i]
        cum[i] = cum[i - 1] + np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    L = cum[-1]
    if L < 1e-9:
        return np.linspace(0, 1, n)
    return cum / L


def _tangent_at(segment: List[Tuple[Tuple[int, int], float]], i: int) -> Tuple[float, float]:
    """Tangent vector (dr, dc) at point i; central difference where possible."""
    n = len(segment)
    (r0, c0), _ = segment[i]
    if i == 0 and n > 1:
        (r1, c1), _ = segment[1]
        return (r1 - r0, c1 - c0)
    if i == n - 1 and n > 1:
        (r1, c1), _ = segment[n - 2]
        return (r0 - r1, c0 - c1)
    if n < 3:
        return (0.0, 0.0)
    (rp, cp), _ = segment[i - 1]
    (rn, cn), _ = segment[i + 1]
    return (rn - rp, cn - cp)


def _normal_2d(dr: float, dc: float) -> Tuple[float, float]:
    """Unit normal: rotate tangent 90° counterclockwise; (dr, dc) -> (-dc, dr) normalized."""
    nrm = np.sqrt(dr * dr + dc * dc)
    if nrm < 1e-9:
        return (0.0, 0.0)
    return (-dc / nrm, dr / nrm)


def perturb_segment(
    segment: List[Tuple[Tuple[int, int], float]],
    A: float,
    omega: float = OMEGA,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Apply sinusoidal normal perturbation to one segment.
    Endpoints (index 0 and -1) unchanged. Interior: row_new = row + A*sin(ω*s)*n_r, col_new = col + A*sin(ω*s)*n_c.
    """
    n = len(segment)
    if n < 3:
        return list(segment)
    s = _arc_length_parameterization(segment)
    out = []
    for i in range(n):
        (r, c), rad = segment[i]
        if i == 0 or i == n - 1:
            out.append(((r, c), rad))
            continue
        tr, tc = _tangent_at(segment, i)
        nr, nc = _normal_2d(tr, tc)
        delta = A * np.sin(omega * s[i])
        r_new = r + delta * nr
        c_new = c + delta * nc
        out.append(((r_new, c_new), rad))
    return out


def perturb_segments(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    A: float,
    omega: float = OMEGA,
    length_weighted: bool = True,
    segment_types: Optional[List[str]] = None,
    alpha: float = ALPHA_ENDPOINT,
    beta: float = BETA_MAIN,
    random_selection_ratio: float = 1.0,
    random_omega: bool = False,
    random_amplitude: bool = False,
    pre_selected_indices: set = None,
    pre_random_omegas: dict = None,
    pre_random_amplitude_scales: dict = None,
) -> List[List[Tuple[Tuple[int, int], float]]]:
    """
    Apply sinusoidal perturbation. Regional weights: A_i = A_global * w_i,
    w_i = alpha if endpoint segment else beta (alpha > beta).
    Endpoints and branch nodes kept fixed; connectivity preserved.
    """
    if not segments:
        return []

    n_segments = len(segments)
    if pre_selected_indices is not None:
        selected_indices = pre_selected_indices
    elif random_selection_ratio < 1.0:
        n_select = max(1, int(n_segments * random_selection_ratio))
        selected_indices = set(np.random.choice(n_segments, n_select, replace=False))
    else:
        selected_indices = set(range(n_segments))

    L_list = None
    L_avg = 1.0
    use_regional = segment_types is not None and len(segment_types) == n_segments
    if not use_regional and length_weighted:
        L_list = [metrics.segment_arc_length(seg) for seg in segments]
        L_avg = sum(L_list) / len(L_list) if L_list else 1.0
        if L_avg < 1e-9:
            L_avg = 1.0

    out = []
    for i, seg in enumerate(segments):
        if i not in selected_indices:
            out.append(list(seg))
            continue
        if use_regional:
            w_i = alpha if segment_types[i] == "endpoint" else beta
            A_seg = A * w_i
        elif length_weighted and L_list:
            A_seg = A * (L_list[i] / L_avg)
        else:
            A_seg = A
        if pre_random_amplitude_scales is not None and i in pre_random_amplitude_scales:
            A_seg = A_seg * pre_random_amplitude_scales[i]
        elif random_amplitude:
            A_seg = A_seg * np.random.uniform(AMPLITUDE_SCALE_LO, AMPLITUDE_SCALE_HI)
        if pre_random_omegas is not None and i in pre_random_omegas:
            omega_seg = pre_random_omegas[i]
        elif random_omega and use_regional:
            # Endpoint: lower freq; main: slightly higher
            if segment_types[i] == "endpoint":
                omega_seg = np.random.uniform(OMEGA_ENDPOINT_LO, OMEGA_ENDPOINT_HI)
            else:
                omega_seg = np.random.uniform(OMEGA_MAIN_LO, OMEGA_MAIN_HI)
        elif random_omega:
            omega_seg = np.random.uniform(OMEGA_ENDPOINT_LO, OMEGA_MAIN_HI)
        else:
            # Fixed: endpoint low-freq large bend, main slightly higher
            omega_seg = OMEGA_ENDPOINT if (use_regional and segment_types[i] == "endpoint") else (OMEGA_MAIN if use_regional else omega)
        out.append(perturb_segment(seg, A_seg, omega_seg))
    return out


def _psi_for_path(
    path_s_boundaries: np.ndarray,
    path_L: float,
    phase_jitter_scale: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Smooth random phase at boundaries; linear interp in between. Returns same length as path_s_boundaries."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(path_s_boundaries)
    psi = rng.uniform(-phase_jitter_scale * np.pi, phase_jitter_scale * np.pi, size=n)
    return psi


def _estimate_density_peak_center(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    sigma: float = 10.0,
) -> Tuple[float, float]:
    """
    Estimate vessel dense center (optic-disc-like) from radius-weighted centerline
    density peak. Thick trunks contribute more than thin capillaries, which helps
    anchor the center at the major vessel hub instead of peripheral branches.
    """
    pts = [(p[0], p[1], rad) for seg in segments for (p, rad) in seg]
    if not pts:
        return 0.0, 0.0
    rows = np.array([p[0] for p in pts], dtype=float)
    cols = np.array([p[1] for p in pts], dtype=float)
    rads = np.array([p[2] for p in pts], dtype=float)
    # Cap extreme values and square for stronger trunk emphasis.
    rad_cap = np.percentile(rads, 95) if len(rads) > 0 else 1.0
    weights = np.clip(rads, 0.0, rad_cap) ** 2
    h = int(np.ceil(rows.max())) + 3
    w = int(np.ceil(cols.max())) + 3
    if h <= 1 or w <= 1:
        if weights.sum() > 1e-9:
            return float(np.average(rows, weights=weights)), float(np.average(cols, weights=weights))
        return float(rows.mean()), float(cols.mean())
    acc = np.zeros((h, w), dtype=float)
    rr = np.clip(np.round(rows).astype(int), 0, h - 1)
    cc = np.clip(np.round(cols).astype(int), 0, w - 1)
    np.add.at(acc, (rr, cc), weights)
    if gaussian_filter is not None:
        score = gaussian_filter(acc, sigma=sigma)
        cr, cc = np.unravel_index(np.argmax(score), score.shape)
        return float(cr), float(cc)
    # Fallback: if scipy is unavailable, use weighted mean.
    if weights.sum() > 1e-9:
        return float(np.average(rows, weights=weights)), float(np.average(cols, weights=weights))
    return float(rows.mean()), float(cols.mean())


def _center_protect_weight(
    row: float,
    col: float,
    center_row: float,
    center_col: float,
    radius: float,
    strength: float,
) -> float:
    """Return attenuation weight in [0, 1]; lower near center."""
    if radius <= 1e-9 or strength <= 1e-9:
        return 1.0
    d = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
    if d >= radius:
        return 1.0
    w = 1.0 - strength * (1.0 - d / radius)
    return float(max(0.0, min(1.0, w)))


def _moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    """1D moving average with reflect padding."""
    if window <= 1 or len(x) <= 2:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def smooth_segments_centerlines(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    window: int = 5,
    iterations: int = 1,
) -> List[List[Tuple[Tuple[int, int], float]]]:
    """
    Smooth centerline coordinates before reconstruction while keeping segment anchors.
    - Preserve first/last point of each segment (branch/endpoint anchors).
    - Smooth only interior points with moving average.
    """
    if iterations <= 0 or window <= 1:
        return [list(seg) for seg in segments]
    out: List[List[Tuple[Tuple[int, int], float]]] = []
    for seg in segments:
        n = len(seg)
        if n < 5:
            out.append(list(seg))
            continue
        rs = np.array([p[0] for (p, _) in seg], dtype=float)
        cs = np.array([p[1] for (p, _) in seg], dtype=float)
        rads = [rad for (_, rad) in seg]
        r0, c0 = rs[0], cs[0]
        r1, c1 = rs[-1], cs[-1]
        for _ in range(iterations):
            rs_s = _moving_average_reflect(rs, window)
            cs_s = _moving_average_reflect(cs, window)
            rs[1:-1] = rs_s[1:-1]
            cs[1:-1] = cs_s[1:-1]
            rs[0], cs[0] = r0, c0
            rs[-1], cs[-1] = r1, c1
        seg_new = [((float(rs[i]), float(cs[i])), rads[i]) for i in range(n)]
        out.append(seg_new)
    return out


def perturb_segments_along_paths(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    A: float,
    K_cycles: float = 1.0,
    phase_jitter_scale: float = 0.6,
    random_seed: Optional[int] = None,
    segment_types: Optional[List[str]] = None,
    alpha_endpoint: float = 0.9,
    beta_main: float = 1.0,
    main_length_gain: float = 0.0,
    center_protect_radius: float = 0.0,
    center_protect_strength: float = 0.0,
    center_protect_point: Optional[Tuple[float, float]] = None,
    amplitude_random_scale: bool = True,
    amplitude_scale_range: Tuple[float, float] = (0.7, 1.3),
    segment_fix_ends: Optional[List[Tuple[bool, bool]]] = None,
) -> List[List[Tuple[Tuple[int, int], float]]]:
    """
    Continuous sinusoidal perturbation along full path; non-periodic (phase drift psi(s)).
    Displacement = A_seg*sin(phi)*n; A_seg = A*(alpha/beta)*(random scale).
    segment_fix_ends: if provided, per-segment (fix_first, fix_last); only fix that end when True. Skeleton endpoint (leaf) can be left unfixed to allow perturbation. None = fix both ends (original behavior).
    """
    rng = np.random.default_rng(random_seed)
    paths, path_L_totals, seg_to_path = path_utils.build_paths(segments)
    if not paths or not seg_to_path:
        return perturb_segments(segments, A, random_omega=False, random_amplitude=False)

    seg_lengths = [metrics.segment_arc_length(seg) for seg in segments]
    path_psi_samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for path_id, path in enumerate(paths):
        L_tot = path_L_totals[path_id] if path_id < len(path_L_totals) else 1.0
        s_boundaries = [0.0]
        for (seg_id, _) in path:
            s_boundaries.append(s_boundaries[-1] + (seg_lengths[seg_id] if seg_id < len(seg_lengths) else 0.0))
        s_boundaries = np.array(s_boundaries)
        psi = _psi_for_path(s_boundaries, L_tot, phase_jitter_scale=phase_jitter_scale, rng=rng)
        path_psi_samples[path_id] = (s_boundaries, psi)

    scale_lo, scale_hi = amplitude_scale_range
    n_seg = len(segments)
    seg_amplitude_scale = rng.uniform(scale_lo, scale_hi, size=n_seg) if amplitude_random_scale else np.ones(n_seg)
    center_row, center_col = (0.0, 0.0)
    if center_protect_radius > 0.0 and center_protect_strength > 0.0:
        if center_protect_point is not None:
            center_row, center_col = center_protect_point
        else:
            center_row, center_col = _estimate_density_peak_center(segments)

    out = []
    for seg_id, seg in enumerate(segments):
        if seg_id not in seg_to_path:
            out.append(list(seg))
            continue
        path_id, s_start, L_path = seg_to_path[seg_id]
        if L_path < 1e-9:
            out.append(list(seg))
            continue
        path = paths[path_id]
        seg_direction = None
        for i, (sid, dr) in enumerate(path):
            if sid == seg_id:
                seg_direction = dr
                break
        if seg_direction is None:
            out.append(list(seg))
            continue
        n_pt = len(seg)
        if n_pt < 3:
            out.append(list(seg))
            continue
        cum = np.zeros(n_pt)
        for i in range(1, n_pt):
            (r0, c0), _ = seg[i - 1]
            (r1, c1), _ = seg[i]
            cum[i] = cum[i - 1] + np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
        L_seg = cum[-1]
        if L_seg < 1e-9:
            out.append(list(seg))
            continue
        use_regional = segment_types is not None and len(segment_types) == len(segments)
        A_seg = A * (alpha_endpoint if (use_regional and segment_types[seg_id] == "endpoint") else beta_main) if use_regional else A
        if use_regional and segment_types[seg_id] == "main" and main_length_gain > 0.0:
            main_boost = 1.0 + main_length_gain * (L_seg / L_path)
            A_seg = A_seg * main_boost
        A_seg = A_seg * seg_amplitude_scale[seg_id]
        fix_first, fix_last = (True, True)
        if segment_fix_ends is not None and seg_id < len(segment_fix_ends):
            fix_first, fix_last = segment_fix_ends[seg_id]
        s_b, psi_b = path_psi_samples[path_id]
        new_seg = []
        for i in range(n_pt):
            (r, c), rad = seg[i]
            if (i == 0 and fix_first) or (i == n_pt - 1 and fix_last):
                new_seg.append(((r, c), rad))
                continue
            if seg_direction == 0:
                s_path = s_start + cum[i]
            else:
                s_path = s_start + (L_seg - cum[i])
            phi_linear = 2.0 * np.pi * K_cycles * (s_path / L_path)
            psi_s = np.interp(s_path, s_b, psi_b)
            phi = phi_linear + psi_s
            tr, tc = _tangent_at(seg, i)
            nr, nc = _normal_2d(tr, tc)
            delta = A_seg * np.sin(phi)
            if center_protect_radius > 0.0 and center_protect_strength > 0.0:
                delta = delta * _center_protect_weight(
                    r, c,
                    center_row, center_col,
                    center_protect_radius,
                    center_protect_strength,
                )
            r_new = r + delta * nr
            c_new = c + delta * nc
            new_seg.append(((r_new, c_new), rad))
        out.append(new_seg)
    return out


# Path sine + VTI closed-loop: bisect amplitude A so achieved T approx VTI_target
PATH_VTI_A_MAX = 80.0
PATH_VTI_TOL = 5e-4
PATH_VTI_MAX_ITER = 50


def apply_path_sine_to_match_vti(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    segment_types: List[str],
    VTI_target: float,
    random_seed: Optional[int] = None,
    A_max: float = PATH_VTI_A_MAX,
    tolerance: float = PATH_VTI_TOL,
    max_iter: int = PATH_VTI_MAX_ITER,
    K_cycles: float = 1.0,
    phase_jitter_scale: float = 0.6,
    alpha_endpoint: float = 0.9,
    beta_main: float = 1.0,
    main_length_gain: float = 0.0,
    center_protect_radius: float = 0.0,
    center_protect_strength: float = 0.0,
    center_protect_point: Optional[Tuple[float, float]] = None,
    amplitude_random_scale: bool = True,
    amplitude_scale_range: Tuple[float, float] = (0.7, 1.3),
    segment_fix_ends: Optional[List[Tuple[bool, bool]]] = None,
) -> Tuple[List[List[Tuple[Tuple[int, int], float]]], float, float]:
    """
    Path-continuous sine + bisect amplitude A so achieved T approx VTI_target.
    segment_fix_ends: per-segment (fix_first, fix_last); only branch-point end fixed; endpoint end can be unfixed to allow perturbation.
    Returns (perturbed_segments, achieved_T, best_A).
    """
    T_baseline = metrics.compute_global_tortuosity(segments)
    if abs(T_baseline - VTI_target) <= tolerance:
        return [list(s) for s in segments], T_baseline, 0.0

    A_lo, A_hi = 0.0, A_max
    best_seg = [list(s) for s in segments]
    best_T = T_baseline
    best_A = 0.0

    for _ in range(max_iter):
        A_mid = (A_lo + A_hi) / 2.0
        seg_pert = perturb_segments_along_paths(
            segments,
            A=A_mid,
            K_cycles=K_cycles,
            phase_jitter_scale=phase_jitter_scale,
            random_seed=random_seed,
            segment_types=segment_types,
            alpha_endpoint=alpha_endpoint,
            beta_main=beta_main,
            main_length_gain=main_length_gain,
            center_protect_radius=center_protect_radius,
            center_protect_strength=center_protect_strength,
            center_protect_point=center_protect_point,
            amplitude_random_scale=amplitude_random_scale,
            amplitude_scale_range=amplitude_scale_range,
            segment_fix_ends=segment_fix_ends,
        )
        T_mid = metrics.compute_global_tortuosity(seg_pert)
        if abs(T_mid - VTI_target) < abs(best_T - VTI_target):
            best_T = T_mid
            best_seg = seg_pert
            best_A = A_mid
        if abs(T_mid - VTI_target) <= tolerance:
            return seg_pert, T_mid, A_mid
        if T_mid < VTI_target:
            A_lo = A_mid
        else:
            A_hi = A_mid
        if A_hi - A_lo < 1e-6:
            break
    return best_seg, best_T, best_A


def _min_radius_over_segments(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> float:
    """Minimum radius over all points; fallback 1.0 if empty."""
    rads = [rad for seg in segments for (_, _), rad in seg]
    return float(min(rads)) if rads else 1.0


def apply_tortuosity_control(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    VTI_target: float,
    segment_types: Optional[List[str]] = None,
    alpha: float = ALPHA_ENDPOINT,
    beta: float = BETA_MAIN,
    tolerance: float = TOLERANCE,
    A_max: float = A_MAX,
    max_iter: int = MAX_ITER,
    random_selection_ratio: float = 1.0,
    random_omega: bool = True,
    random_amplitude: bool = True,
    random_seed: int = None,
    use_fixed_amplitude: bool = False,
    fixed_amplitude: Optional[float] = None,
    use_raw_amplitude: bool = False,
) -> Tuple[List[List[Tuple[Tuple[int, int], float]]], float]:
    """
    If use_fixed_amplitude: use global A (no binary search). If use_raw_amplitude: A = fixed_amplitude directly;
    else A = fixed_amplitude or 0.6*min_radius, scaled by (VTI_target / T_baseline). Otherwise: binary search for VTI.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    T_current = metrics.compute_global_tortuosity(segments)
    n_segments = len(segments)
    if random_selection_ratio < 1.0:
        n_select = max(1, int(n_segments * random_selection_ratio))
        selected_indices = set(np.random.choice(n_segments, n_select, replace=False))
    else:
        selected_indices = set(range(n_segments))

    use_regional = segment_types is not None and len(segment_types) == n_segments
    random_omegas = None
    random_amplitude_scales = None
    if random_omega:
        if use_regional:
            random_omegas = {
                i: np.random.uniform(OMEGA_ENDPOINT_LO, OMEGA_ENDPOINT_HI) if segment_types[i] == "endpoint"
                else np.random.uniform(OMEGA_MAIN_LO, OMEGA_MAIN_HI)
                for i in selected_indices
            }
        else:
            random_omegas = {i: np.random.uniform(OMEGA_ENDPOINT_LO, OMEGA_MAIN_HI) for i in selected_indices}
    if random_amplitude:
        random_amplitude_scales = {i: np.random.uniform(AMPLITUDE_SCALE_LO, AMPLITUDE_SCALE_HI) for i in selected_indices}

    def do_perturb(A: float):
        return perturb_segments(
            segments, A,
            length_weighted=(segment_types is None or len(segment_types) != n_segments),
            segment_types=segment_types,
            alpha=alpha,
            beta=beta,
            random_selection_ratio=1.0,
            random_omega=False,
            random_amplitude=False,
            pre_selected_indices=selected_indices,
            pre_random_omegas=random_omegas,
            pre_random_amplitude_scales=random_amplitude_scales,
        )

    # Fixed-amplitude mode: no binary search
    if use_fixed_amplitude:
        if use_raw_amplitude and fixed_amplitude is not None:
            A = min(float(fixed_amplitude), A_max)
        else:
            min_r = _min_radius_over_segments(segments)
            A_base = fixed_amplitude if fixed_amplitude is not None else (0.6 * min_r)
            if T_current < 1e-9:
                T_current = 1.0
            A = A_base * (VTI_target / T_current)
            A = min(A, A_max)
        new_seg = do_perturb(A)
        T_achieved = metrics.compute_global_tortuosity(new_seg)
        return new_seg, T_achieved

    if abs(T_current - VTI_target) <= tolerance:
        return [list(s) for s in segments], T_current

    A_lo, A_hi = 0.0, A_max
    best_segments = [list(s) for s in segments]
    best_T = T_current
    best_A = 0.0

    for _ in range(max_iter):
        A_mid = (A_lo + A_hi) / 2.0
        new_seg = do_perturb(A_mid)
        T_mid = metrics.compute_global_tortuosity(new_seg)
        if abs(T_mid - VTI_target) < abs(best_T - VTI_target):
            best_T = T_mid
            best_segments = new_seg
            best_A = A_mid
        if abs(T_mid - VTI_target) <= tolerance:
            return new_seg, T_mid
        if T_mid < VTI_target:
            A_lo = A_mid
        else:
            A_hi = A_mid
        if A_hi - A_lo < 1e-6:
            break

    if best_T < VTI_target and best_A >= A_max * 0.99:
        import warnings
        warnings.warn(
            f"Tortuosity: achieved {best_T:.4f} < target {VTI_target:.4f} (A hit A_max={A_max})."
        )
    return best_segments, best_T
