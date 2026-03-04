"""
Stage 5: Box-counting fractal dimension (validation only, not used for control).

On skeleton image only: count covering boxes at multiple scales, fit log(N) vs log(1/ε), slope = D.
"""

import numpy as np
from typing import List, Optional


def box_count(skeleton: np.ndarray, box_size: int) -> int:
    """
    Count minimum number of boxes of side length box_size needed to cover all skeleton pixels.
    Grid aligned: partition image into non-overlapping boxes of size box_size x box_size;
    count boxes that contain at least one skeleton pixel.
    """
    h, w = skeleton.shape
    if box_size < 1:
        box_size = 1
    n_r = (h + box_size - 1) // box_size
    n_c = (w + box_size - 1) // box_size
    count = 0
    for i in range(n_r):
        for j in range(n_c):
            r0, r1 = i * box_size, min((i + 1) * box_size, h)
            c0, c1 = j * box_size, min((j + 1) * box_size, w)
            if np.any(skeleton[r0:r1, c0:c1] > 0):
                count += 1
    return count


def fractal_dimension_box_counting(
    skeleton: np.ndarray,
    box_sizes: Optional[List[int]] = None,
) -> float:
    """
    Compute fractal dimension via box-counting: D = -d(log N)/d(log ε) with ε = box_size.
    Using log(1/ε) = -log(ε): log N ~ D * log(1/ε) + const  =>  slope of log N vs log(1/ε) = D.
    So we use x = log(box_size), y = log(N); then D = -slope of y vs x, i.e. slope of log N vs log(1/box_size).
    Equivalently: x = log(1/box_size), y = log(N), then D = slope.
    """
    if box_sizes is None:
        # Multiple scales: powers of 2 from 2 up to ~1/4 of min dimension
        min_side = min(skeleton.shape[0], skeleton.shape[1])
        max_box = max(2, min_side // 4)
        box_sizes = []
        b = 2
        while b <= max_box:
            box_sizes.append(b)
            b *= 2
        if not box_sizes:
            box_sizes = [2, 4]
    if not box_sizes or np.sum(skeleton) == 0:
        return 0.0

    x = np.log(1.0 / np.array(box_sizes, dtype=np.float64))
    y = np.array([np.log(max(1, box_count(skeleton, s))) for s in box_sizes], dtype=np.float64)
    # D = slope of y vs x (log N vs log(1/ε))
    if len(x) < 2:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)
