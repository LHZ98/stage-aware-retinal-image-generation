"""
Stage 1: Skeletonization and radius extraction for binary retinal vessel masks.

- Skeletonize with skimage.morphology.skeletonize.
- Radius at each skeleton pixel = EDT value (scipy.ndimage.distance_transform_edt).
- Graph: branch point = degree >= 3, endpoint = degree == 1 (8-connected).
- Segments between branch points, endpoint-branch, endpoint-endpoint.
- Ignore segments shorter than MIN_SEGMENT_LENGTH pixels.
- Per segment: coords (r,c), per-point radius, type "endpoint" or "main".
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import List, Tuple

# Minimum segment length (pixels) to keep (prompt: 5)
MIN_SEGMENT_LENGTH = 5


def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize binary vessel mask.
    mask: (H, W), 1 = vessel, 0 = background.
    Returns: binary skeleton (H, W), 1 = skeleton pixel.
    """
    return skeletonize(mask.astype(bool)).astype(np.uint8)


def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance transform: value at each pixel = distance to nearest background.
    Radius at skeleton pixel = this value (skeleton lies on medial axis).
    """
    return ndimage.distance_transform_edt(mask > 0)


def get_branch_points(skeleton: np.ndarray) -> np.ndarray:
    """
    Branch points = skeleton pixels with >= 3 neighbors (8-connectivity).
    Returns: boolean array (H, W), True at branch points.
    """
    from scipy.ndimage import convolve
    # 8-neighborhood kernel: center 0, neighbors 1
    kernel = np.ones((3, 3), dtype=np.int32)
    kernel[1, 1] = 0
    neighbor_count = convolve(skeleton.astype(np.int32), kernel, mode="constant", cval=0)
    return (skeleton > 0) & (neighbor_count >= 3)


def get_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Endpoints = skeleton pixels with exactly 1 neighbor (8-connectivity)."""
    from scipy.ndimage import convolve
    kernel = np.ones((3, 3), dtype=np.int32)
    kernel[1, 1] = 0
    neighbor_count = convolve(skeleton.astype(np.int32), kernel, mode="constant", cval=0)
    return (skeleton > 0) & (neighbor_count == 1)


def _neighbors_8(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Return 8-connected neighbor (r,c) indices within [0,h)x[0,w)."""
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def _extract_segment(
    skeleton: np.ndarray,
    dist: np.ndarray,
    branch_or_end: np.ndarray,
    start: Tuple[int, int],
    next_from_start: Tuple[int, int],
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Walk from start along skeleton until we hit another branch/endpoint.
    Returns list of ((r, c), radius).
    Order: start -> ... -> end (branch or endpoint).
    """
    h, w = skeleton.shape
    path = [start]
    prev = start
    cur = next_from_start
    while True:
        path.append(cur)
        if branch_or_end[cur[0], cur[1]]:
            break
        # next step: neighbor of cur that is skeleton and not prev
        cands = _neighbors_8(cur[0], cur[1], h, w)
        next_cur = None
        for n in cands:
            if n == prev:
                continue
            if skeleton[n[0], n[1]]:
                next_cur = n
                break
        if next_cur is None:
            break
        prev, cur = cur, next_cur
    return [(p, float(dist[p[0], p[1]])) for p in path]


def extract_segments_and_types(
    skeleton: np.ndarray,
    dist: np.ndarray,
    min_length: int = MIN_SEGMENT_LENGTH,
) -> Tuple[List[List[Tuple[Tuple[int, int], float]]], List[str]]:
    """
    Split skeleton into segments between branch points and endpoints.
    Each segment: list of ((r, c), radius). Type: "endpoint" or "main".
    Segments with length < min_length are dropped.
    Returns (segments, segment_types).
    """
    branch = get_branch_points(skeleton)
    endpts = get_endpoints(skeleton)
    branch_or_end = branch | endpts
    h, w = skeleton.shape
    seen_segment = set()
    segments = []
    segment_types = []

    def add_segment(start: Tuple[int, int], next_from_start: Tuple[int, int]) -> None:
        seg = _extract_segment(skeleton, dist, branch_or_end, start, next_from_start)
        if len(seg) < 2:
            return
        end0, end1 = seg[0][0], seg[-1][0]
        key = (min(end0, end1), max(end0, end1))
        if key in seen_segment:
            return
        arc = 0.0
        for i in range(len(seg) - 1):
            r0, c0 = seg[i][0]
            r1, c1 = seg[i + 1][0]
            arc += np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
        if arc < min_length:
            return
        is_end0 = bool(endpts[end0[0], end0[1]])
        is_end1 = bool(endpts[end1[0], end1[1]])
        seg_type = "endpoint" if (is_end0 or is_end1) else "main"
        seen_segment.add(key)
        segments.append(seg)
        segment_types.append(seg_type)

    for r in range(h):
        for c in range(w):
            if not skeleton[r, c]:
                continue
            start = (r, c)
            if not branch_or_end[r, c]:
                continue
            for nr, nc in _neighbors_8(r, c, h, w):
                if not skeleton[nr, nc]:
                    continue
                add_segment(start, (nr, nc))

    return segments, segment_types


def extract_skeleton_and_segments(
    mask: np.ndarray,
    min_length: int = MIN_SEGMENT_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[Tuple[int, int], float]]], List[str]]:
    """
    Full Stage 1 pipeline.
    Returns:
        skeleton: (H, W) binary
        dist: (H, W) distance transform (radius map)
        segments: list of segments; each segment = list of ((r,c), radius)
        segment_types: list of "endpoint" | "main" per segment
    """
    skeleton = compute_skeleton(mask)
    dist = compute_distance_transform(mask)
    segments, segment_types = extract_segments_and_types(skeleton, dist, min_length=min_length)
    return skeleton, dist, segments, segment_types
