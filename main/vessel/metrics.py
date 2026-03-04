"""
Stage 2: Length-weighted global tortuosity and caliber metrics.

Per segment i:
  L_i = arc length (sum of Euclidean distances between consecutive points)
  D_i = chord length (Euclidean distance between endpoints)
  tau_i = L_i / D_i  (segment tortuosity; D_i ~ 0 handled)
  r̄_i = mean radius along segment

Global (length-weighted):
  T = sum(L_i * tau_i) / sum(L_i)   [global tortuosity]
  C = sum(L_i * r̄_i) / sum(L_i)     [global caliber]
"""

import numpy as np
from typing import List, Tuple

# Avoid division by zero when chord length is negligible
CHORD_EPS = 1e-6


def segment_arc_length(segment: List[Tuple[Tuple[int, int], float]]) -> float:
    """L_i = sum of Euclidean distances between consecutive centerline points."""
    if len(segment) < 2:
        return 0.0
    total = 0.0
    for i in range(len(segment) - 1):
        (r0, c0), _ = segment[i]
        (r1, c1), _ = segment[i + 1]
        total += np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    return total


def segment_chord_length(segment: List[Tuple[Tuple[int, int], float]]) -> float:
    """D_i = Euclidean distance between first and last point."""
    if len(segment) < 2:
        return 0.0
    (r0, c0), _ = segment[0]
    (r1, c1), _ = segment[-1]
    return np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)


def segment_vti(segment: List[Tuple[Tuple[int, int], float]]) -> float:
    """VTI_i = L_i / D_i. If D_i < CHORD_EPS, return 1.0 (no curvature)."""
    L = segment_arc_length(segment)
    D = segment_chord_length(segment)
    if D < CHORD_EPS:
        return 1.0
    return L / D


def segment_mean_radius(segment: List[Tuple[Tuple[int, int], float]]) -> float:
    """r̄_i = average radius along that segment."""
    if not segment:
        return 0.0
    return sum(r for _, r in segment) / len(segment)


def compute_global_tortuosity(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> float:
    """
    T_global = sum(L_i * VTI_i) / sum(L_i).
    Length-weighted average of segment tortuosities.
    """
    if not segments:
        return 1.0
    total_L = 0.0
    weighted_total = 0.0
    for seg in segments:
        L_i = segment_arc_length(seg)
        if L_i < 1e-9:
            continue
        VTI_i = segment_vti(seg)
        total_L += L_i
        weighted_total += L_i * VTI_i
    if total_L < 1e-9:
        return 1.0
    return weighted_total / total_L


def compute_global_width(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> float:
    """
    W_global = sum(L_i * r̄_i) / sum(L_i).
    Length-weighted average radius. Alias: use compute_global_caliber for prompt naming.
    """
    return compute_global_caliber(segments)


def compute_global_caliber(
    segments: List[List[Tuple[Tuple[int, int], float]]],
) -> float:
    """
    C = sum(L_i * r̄_i) / sum(L_i).
    Global caliber: length-weighted average radius.
    """
    if not segments:
        return 0.0
    total_L = 0.0
    weighted_total = 0.0
    for seg in segments:
        L_i = segment_arc_length(seg)
        if L_i < 1e-9:
            continue
        r_bar = segment_mean_radius(seg)
        total_L += L_i
        weighted_total += L_i * r_bar
    if total_L < 1e-9:
        return 0.0
    return weighted_total / total_L
