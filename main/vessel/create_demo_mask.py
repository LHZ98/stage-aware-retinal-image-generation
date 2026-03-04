"""
Create a simple synthetic vessel mask for testing experiment without real data.

Usage (from repo root):
  python main/vessel/create_demo_mask.py
  python -m main.vessel.experiment --mask main/vessel/demo_mask.npy --out_dir ./vessel_out

Output: main/vessel/demo_mask.npy (H=128, W=128 binary mask)
"""

import numpy as np
import os

def create_demo_mask(height=128, width=128):
    """Draw a few simple vessels: polylines + radius, then binarize."""
    mask = np.zeros((height, width), dtype=np.uint8)
    # Centerline + radius (r, c, radius)
    vessels = [
        # Horizontal
        (np.array([20, 20, 20]), np.array([20, 64, 108]), np.array([4, 5, 4])),
        # Vertical
        (np.array([30, 64, 98]), np.array([64, 64, 64]), np.array([3, 4, 3])),
        # Diagonal
        (np.array([80, 90, 100]), np.array([30, 50, 70]), np.array([3, 4, 3])),
    ]
    for rr, cc, rad in vessels:
        for i in range(len(rr)):
            r, c, R = int(rr[i]), int(cc[i]), max(1, int(rad[i]))
            for dr in range(-R - 1, R + 2):
                for dc in range(-R - 1, R + 2):
                    if dr * dr + dc * dc <= R * R:
                        rn, cn = r + dr, c + dc
                        if 0 <= rn < height and 0 <= cn < width:
                            mask[rn, cn] = 1
    return mask


if __name__ == "__main__":
    d = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(d, "demo_mask.npy")
    mask = create_demo_mask()
    np.save(out_path, mask)
    print("Saved:", out_path, "shape:", mask.shape, "vessel pixels:", np.sum(mask))
