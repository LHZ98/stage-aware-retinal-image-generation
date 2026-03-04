"""
Stage 4: Caliber control by scaling radius and reconstructing mask from disks.

- Scale: r_new = lambda * r_original.
- Enforce Murray's law at each branch: r_parent^3 = r_child1^3 + r_child2^3.
  If violated beyond tolerance, project child radii to nearest feasible solution.
- Reconstruct: draw disk(center, radius) at each skeleton point; merge (union).
- Topology preserved; no holes; connectivity preserved.
"""

import numpy as np
from typing import List, Tuple

# Minimum disk radius when drawing (pixels) for numerical stability
MIN_DISK_RADIUS = 0.5
# Murray's law tolerance (relative)
MURRAY_TOLERANCE = 1e-3


def scale_segment_radii(
    segment: List[Tuple[Tuple[int, int], float]],
    alpha: float,
) -> List[Tuple[Tuple[int, int], float]]:
    """Return segment with radii scaled: new_r = α * original_r. Coords unchanged."""
    return [((p[0], p[1]), alpha * r) for p, r in segment]


def scale_segments_radii(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    alpha: float,
) -> List[List[Tuple[Tuple[int, int], float]]]:
    """Scale radius at every skeleton point by α."""
    return [scale_segment_radii(seg, alpha) for seg in segments]


def _draw_disk(
    mask: np.ndarray,
    center_row: float,
    center_col: float,
    radius: float,
) -> None:
    """Draw filled disk (radius in pixels) into mask; in-place. radius >= MIN_DISK_RADIUS."""
    r = max(radius, MIN_DISK_RADIUS)
    R = int(np.ceil(r)) + 1
    h, w = mask.shape
    cr, cc = int(round(center_row)), int(round(center_col))
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            rr, ccc = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= ccc < w:
                if dr * dr + dc * dc <= r * r:
                    mask[rr, ccc] = 1


def _draw_capsule(
    mask: np.ndarray,
    r0: float, c0: float, rad0: float,
    r1: float, c1: float, rad1: float,
) -> None:
    """
    Draw a capsule between (r0,c0) and (r1,c1) with radii rad0, rad1.
    Sample points along the segment and draw a disk at each; ensures no gap between consecutive centers.
    """
    dist = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    if dist < 1e-9:
        _draw_disk(mask, r0, c0, max(rad0, rad1, MIN_DISK_RADIUS))
        return
    # Step ~0.5 pixel so disks overlap along the segment
    n_steps = max(2, int(np.ceil(dist / 0.5)))
    for k in range(n_steps + 1):
        t = k / n_steps
        r = (1 - t) * r0 + t * r1
        c = (1 - t) * c0 + t * c1
        rad = (1 - t) * rad0 + t * rad1
        _draw_disk(mask, r, c, rad)


def reconstruct_mask_from_segments(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct binary vessel mask from segments:
    - At each (row, col) with radius r, draw a disk of radius r.
    - Between every consecutive pair of points in a segment, draw a capsule (filled tube)
      so that wiggly centerlines do not produce gaps/breakpoints.
    Returns (H, W) array, vessel=1, background=0.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for seg in segments:
        for (r, c), rad in seg:
            _draw_disk(mask, r, c, rad)
        for i in range(len(seg) - 1):
            (r0, c0), rad0 = seg[i]
            (r1, c1), rad1 = seg[i + 1]
            _draw_capsule(mask, r0, c0, rad0, r1, c1, rad1)
    return mask


def _branch_point_radius_map(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> dict:
    """Map (r, c) -> list of (seg_idx, end_0_or_minus1, radius). Branch points have >= 3 entries."""
    from collections import defaultdict
    point_to_radii = defaultdict(list)
    for seg_idx, seg in enumerate(segments):
        if not seg:
            continue
        (rc0, r0) = seg[0]
        point_to_radii[rc0].append((seg_idx, 0, r0))
        if len(seg) > 1:
            (rc1, r1) = seg[-1]
            if rc1 != rc0:
                point_to_radii[rc1].append((seg_idx, -1, r1))
    return dict(point_to_radii)


def _project_murray(r_parent: float, r_c1: float, r_c2: float) -> Tuple[float, float]:
    """
    Project (r_c1, r_c2) onto r_parent^3 = r_c1^3 + r_c2^3 minimizing deviation.
    In (u,v)=(r_c1^3, r_c2^3): project (u_s, v_s) onto u+v = K, then r_c1 = u^(1/3), r_c2 = v^(1/3).
    """
    K = r_parent ** 3
    u_s = r_c1 ** 3
    v_s = r_c2 ** 3
    # Nearest on line u + v = K: u_proj = (K + u_s - v_s) / 2, v_proj = K - u_proj
    u_proj = (K + u_s - v_s) / 2.0
    u_proj = max(0.0, min(K, u_proj))
    v_proj = K - u_proj
    if v_proj < 0:
        v_proj = 0.0
        u_proj = K
    r_c1_new = (u_proj ** (1.0 / 3.0)) if u_proj > 0 else 0.0
    r_c2_new = (v_proj ** (1.0 / 3.0)) if v_proj > 0 else 0.0
    return r_c1_new, r_c2_new


def enforce_murray_at_branches(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    tolerance: float = MURRAY_TOLERANCE,
) -> List[List[Tuple[Tuple[int, int], float]]]:
    """
    Enforce Murray's law at each branch: r_parent^3 = r_child1^3 + r_child2^3.
    For each bifurcation, assign largest radius as parent; project child radii to feasible.
    Returns new segments (copy with radii updated at branch endpoints).
    """
    out = [[((p[0], p[1]), r) for p, r in seg] for seg in segments]
    point_to_radii = _branch_point_radius_map(out)

    for (r, c), entries in point_to_radii.items():
        if len(entries) < 3:
            continue
        # Bifurcation: use first 3 (or take 3 with largest radii)
        entries_sorted = sorted(entries, key=lambda x: x[2], reverse=True)
        (p_idx, p_end, r_p), (c1_idx, c1_end, r_c1), (c2_idx, c2_end, r_c2) = entries_sorted[0], entries_sorted[1], entries_sorted[2]
        K = r_p ** 3
        sum_cubes = r_c1 ** 3 + r_c2 ** 3
        if abs(K - sum_cubes) <= tolerance * (K + 1e-12):
            continue
        r_c1_new, r_c2_new = _project_murray(r_p, r_c1, r_c2)
        # Write back: update radius at (p_end, c1_end, c2_end) in out
        if c1_end == 0:
            coord = out[c1_idx][0][0]
            out[c1_idx][0] = (coord, r_c1_new)
        else:
            coord = out[c1_idx][-1][0]
            out[c1_idx][-1] = (coord, r_c1_new)
        if c2_end == 0:
            coord = out[c2_idx][0][0]
            out[c2_idx][0] = (coord, r_c2_new)
        else:
            coord = out[c2_idx][-1][0]
            out[c2_idx][-1] = (coord, r_c2_new)
    return out


def apply_width_control(
    segments: List[List[Tuple[Tuple[int, int], float]]],
    alpha: float,
    shape: Tuple[int, int],
    enforce_murray: bool = True,
) -> np.ndarray:
    """
    Scale all radii by lambda (alpha); optionally enforce Murray's law at branches; reconstruct mask.
    Returns binary mask (H, W).
    """
    scaled = scale_segments_radii(segments, alpha)
    if enforce_murray:
        scaled = enforce_murray_at_branches(scaled)
    return reconstruct_mask_from_segments(scaled, shape)
