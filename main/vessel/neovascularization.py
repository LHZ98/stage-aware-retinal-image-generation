"""
Neovascularization: new vessels spawned at random points along segments (mid-segment or endpoint with lower prob e.g. 0.35).
New vessels are curved (sinusoidal normal perturbation), not straight. Length 20–100 px, radius >= min_radius.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict


def _arc_lengths(segment: List[Tuple[Tuple[int, int], float]]) -> np.ndarray:
    """Cumulative arc length at each point; length n."""
    n = len(segment)
    cum = np.zeros(n)
    for i in range(1, n):
        (r0, c0), _ = segment[i - 1]
        (r1, c1), _ = segment[i]
        cum[i] = cum[i - 1] + np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    return cum


def _point_along_segment(
    segment: List[Tuple[Tuple[int, int], float]],
    frac: float,
) -> Tuple[Tuple[float, float], float, Tuple[float, float]]:
    """Interpolate at fraction frac of arc length. Returns (r,c), radius, tangent (unit)."""
    n = len(segment)
    if n < 2:
        (r, c), rad = segment[0]
        return (float(r), float(c)), float(rad), (1.0, 0.0)
    cum = _arc_lengths(segment)
    total = cum[-1]
    if total < 1e-9:
        (r, c), rad = segment[0]
        return (float(r), float(c)), float(rad), (1.0, 0.0)
    s = frac * total
    i = 0
    while i < n - 1 and cum[i + 1] < s:
        i += 1
    if i >= n - 1:
        i = n - 2
    (r0, c0), rad0 = segment[i]
    (r1, c1), rad1 = segment[i + 1]
    seg_len = cum[i + 1] - cum[i]
    t = (s - cum[i]) / seg_len if seg_len > 1e-9 else 0.0
    t = max(0, min(1, t))
    rr = (1 - t) * r0 + t * r1
    cc = (1 - t) * c0 + t * c1
    rad = (1 - t) * rad0 + t * rad1
    dr, dc = r1 - r0, c1 - c0
    nrm = np.sqrt(dr * dr + dc * dc)
    tangent = (dr / nrm, dc / nrm) if nrm > 1e-9 else (1.0, 0.0)
    return (rr, cc), rad, tangent


def _normal_2d(tx: float, ty: float) -> Tuple[float, float]:
    """Unit normal: rotate tangent 90° counterclockwise; (-ty, tx)."""
    nrm = np.sqrt(tx * tx + ty * ty)
    if nrm < 1e-9:
        return (1.0, 0.0)
    return (-ty / nrm, tx / nrm)


def _tangent_at_end(
    segment: List[Tuple[Tuple[int, int], float]],
    at_start: bool,
) -> Tuple[float, float]:
    """Outward tangent at the segment end (at_start=True -> index 0)."""
    n = len(segment)
    if n < 2:
        return (1.0, 0.0)
    if at_start:
        (r0, c0), _ = segment[0]
        (r1, c1), _ = segment[1]
        dr, dc = r0 - r1, c0 - c1
    else:
        (r0, c0), _ = segment[-1]
        (r1, c1), _ = segment[-2]
        dr, dc = r0 - r1, c0 - c1
    nrm = np.sqrt(dr * dr + dc * dc)
    if nrm < 1e-9:
        return (1.0, 0.0)
    return (dr / nrm, dc / nrm)


def _terminal_endpoints(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> List[Tuple[int, int, int, bool]]:
    """Return list of (seg_idx, end_0_or_minus1, is_start) for each terminal tip.
    Terminal = (r,c) that appears as endpoint of exactly one segment."""
    point_to_ends = defaultdict(list)  # (r,c) -> [(seg_idx, at_start), ...]
    for seg_idx, seg in enumerate(segments):
        if not seg:
            continue
        (r0, c0), _ = seg[0]
        (r1, c1), _ = seg[-1]
        point_to_ends[(r0, c0)].append((seg_idx, True))
        if (r1, c1) != (r0, c0):
            point_to_ends[(r1, c1)].append((seg_idx, False))
    terminals = []
    for (r, c), ends in point_to_ends.items():
        if len(ends) == 1:
            seg_idx, at_start = ends[0]
            terminals.append((seg_idx, -1 if at_start else 0, at_start))  # end index 0 or -1
    return terminals


def _append_new_vessel_curved(
    out: list,
    start_r: float,
    start_c: float,
    direction: Tuple[float, float],
    length: float,
    r_small: float,
    h: int,
    w: int,
    curve_amplitude: float = 4.0,
    curve_omega: float = 1.5 * np.pi,
    phase_jitter_scale: float = 0.6,
) -> None:
    """Append one new vessel: grow along direction with curved path (phase jitter, non-periodic)."""
    n_pts = max(5, int(round(length / 1.2)))
    step = length / max(1, n_pts)
    dx, dy = direction
    px, py = -dy, dx
    nrm = np.sqrt(px * px + py * py)
    if nrm > 1e-9:
        px, py = px / nrm, py / nrm
    else:
        px, py = 1.0, 0.0
    # Phase jitter: sample 3 random phases along arc, linear interpolate so bend is non-periodic
    s_samples = np.array([0.0, 0.4, 0.7, 1.0])
    psi_samples = np.random.uniform(-phase_jitter_scale * np.pi, phase_jitter_scale * np.pi, size=4)
    new_pts = [((start_r, start_c), r_small)]
    for i in range(1, n_pts + 1):
        t = i * step
        s = t / length if length > 1e-9 else 0.0
        base_r = start_r + t * dx
        base_c = start_c + t * dy
        psi_s = float(np.interp(s, s_samples, psi_samples))
        phi = curve_omega * s + psi_s
        offset = curve_amplitude * np.sin(phi)
        rr = base_r + offset * px
        cc = base_c + offset * py
        rr = max(0, min(h - 1e-6, rr))
        cc = max(0, min(w - 1e-6, cc))
        new_pts.append(((rr, cc), r_small))
    if len(new_pts) >= 2:
        out.append(new_pts)


def add_small_vessels_at_endpoints(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    shape: Tuple[int, int],
    length_range: Tuple[float, float] = (20, 150),
    max_new_vessel_length: Optional[float] = None,
    radius_scale_range: Tuple[float, float] = (0.32, 0.52),
    min_radius: float = 4.0,
    prob_segment: float = 0.1,
    prob_at_endpoint: float = 0.35,
    min_segment_length_for_interior: float = 10.0,
    curve_amplitude_range: Tuple[float, float] = (2.0, 6.0),
    phase_jitter_scale: float = 0.6,
    random_seed: Optional[int] = None,
) -> List[List[Tuple[Tuple[int, int], float]]]:
    """
    Per segment: with prob_segment decide whether to add neo; if so, with prob_at_endpoint at terminal else at mid-segment.
    If preferred position is unavailable (e.g. no terminal), fall back to the other. max_new_vessel_length caps neo length (px). Curved (phase jitter).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    out = [list(seg) for seg in segments]
    h, w = shape
    min_len = float(length_range[0])
    max_len = float(length_range[1])
    if max_new_vessel_length is not None:
        max_len = min(max_len, float(max_new_vessel_length))
    if max_len < min_len:
        max_len = min_len
    terminals = _terminal_endpoints(segments)
    terminal_map = defaultdict(list)  # seg_idx -> [at_start(bool), ...]
    for seg_idx, _, at_start in terminals:
        terminal_map[seg_idx].append(at_start)

    def add_one_curved(r0: float, c0: float, dx: float, dy: float, length: float, r_small: float) -> None:
        amp = np.random.uniform(curve_amplitude_range[0], curve_amplitude_range[1])
        omega = np.random.uniform(1.0 * np.pi, 2.0 * np.pi)
        _append_new_vessel_curved(
            out, r0, c0, (dx, dy), length, r_small, h, w,
            curve_amplitude=amp, curve_omega=omega, phase_jitter_scale=phase_jitter_scale,
        )

    # Per segment: with prob_segment add one neo; place by prob_at_endpoint at endpoint else mid-segment.
    for seg_idx, seg in enumerate(segments):
        if np.random.rand() > prob_segment:
            continue
        if len(seg) < 2:
            continue
        seg_out = out[seg_idx]
        choose_endpoint = np.random.rand() < prob_at_endpoint
        can_endpoint = seg_idx in terminal_map and len(terminal_map[seg_idx]) > 0
        cum = _arc_lengths(seg)
        total_len = cum[-1]
        can_interior = total_len >= min_segment_length_for_interior

        # If preferred position unavailable, fall back to the other.
        if choose_endpoint and can_endpoint:
            at_start = bool(terminal_map[seg_idx][np.random.randint(len(terminal_map[seg_idx]))])
            tip_pt, tip_radius = seg_out[0] if at_start else seg_out[-1]
            tr, tc = _tangent_at_end(seg_out, at_start)
            angle = np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = tr * cos_a - tc * sin_a
            dy = tr * sin_a + tc * cos_a
            r0, c0 = float(tip_pt[0]), float(tip_pt[1])
            length = np.random.uniform(min_len, max_len)
            radius_scale = np.random.uniform(radius_scale_range[0], radius_scale_range[1])
            r_small = max(min_radius, tip_radius * radius_scale)
            add_one_curved(r0, c0, dx, dy, length, r_small)
            continue

        if (not choose_endpoint) and can_interior:
            frac = np.random.uniform(0.2, 0.8)
            (r0, c0), local_radius, tangent = _point_along_segment(seg_out, frac)
            nr, nc = _normal_2d(tangent[0], tangent[1])
            sign = 1 if np.random.rand() > 0.5 else -1
            dx, dy = sign * nr, sign * nc
            length = np.random.uniform(min_len, max_len)
            radius_scale = np.random.uniform(radius_scale_range[0], radius_scale_range[1])
            r_small = max(min_radius, local_radius * radius_scale)
            add_one_curved(r0, c0, dx, dy, length, r_small)
            continue

        if can_endpoint:
            at_start = bool(terminal_map[seg_idx][np.random.randint(len(terminal_map[seg_idx]))])
            tip_pt, tip_radius = seg_out[0] if at_start else seg_out[-1]
            tr, tc = _tangent_at_end(seg_out, at_start)
            angle = np.random.uniform(-0.4 * np.pi, 0.4 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            dx = tr * cos_a - tc * sin_a
            dy = tr * sin_a + tc * cos_a
            r0, c0 = float(tip_pt[0]), float(tip_pt[1])
            length = np.random.uniform(min_len, max_len)
            radius_scale = np.random.uniform(radius_scale_range[0], radius_scale_range[1])
            r_small = max(min_radius, tip_radius * radius_scale)
            add_one_curved(r0, c0, dx, dy, length, r_small)
            continue

        if can_interior:
            frac = np.random.uniform(0.2, 0.8)
            (r0, c0), local_radius, tangent = _point_along_segment(seg_out, frac)
            nr, nc = _normal_2d(tangent[0], tangent[1])
            sign = 1 if np.random.rand() > 0.5 else -1
            dx, dy = sign * nr, sign * nc
            length = np.random.uniform(min_len, max_len)
            radius_scale = np.random.uniform(radius_scale_range[0], radius_scale_range[1])
            r_small = max(min_radius, local_radius * radius_scale)
            add_one_curved(r0, c0, dx, dy, length, r_small)

    return out
