#!/usr/bin/env python3
"""
Draw 2x2 segment extraction visualization:
1. Original Mask (black bg, white vessels)  2. Skeleton  3. Branch Points & Endpoints  4. All Segments.
Usage (from repo root):
  python main/mask_generation/plot_segment_visualization.py \\
    --mask /path/to/mask.png \\
    --out main/mask_generation/visual/1b398c0494d1_segments.png \\
    --dpi 200 --min_length 5
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root
if __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _root)
    __package__ = "main.mask_generation"

from main.vessel.skeleton_utils import (
    compute_skeleton,
    compute_distance_transform,
    get_branch_points,
    get_endpoints,
    extract_skeleton_and_segments,
)


def load_mask(path: str) -> np.ndarray:
    """Load binary mask (H, W), 1 = vessel, 0 = background."""
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


# Extended color palette for many segments (repeat if needed)
def _segment_colors(n: int):
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def main():
    p = argparse.ArgumentParser(description="Segment extraction 2x2 visualization")
    p.add_argument("--mask", type=str, required=True, help="Path to binary vessel mask image")
    p.add_argument("--out", type=str, default=None, help="Output PNG path; default: visual/<stem>_segments.png")
    p.add_argument("--min_length", type=int, default=5, help="Min segment length (pixels)")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI for higher resolution")
    p.add_argument("--fig_width", type=float, default=12, help="Figure width in inches")
    args = p.parse_args()

    mask_path = os.path.abspath(args.mask)
    if not os.path.isfile(mask_path):
        print("Mask file not found:", mask_path)
        sys.exit(1)

    mask = load_mask(mask_path)
    case_id = os.path.splitext(os.path.basename(mask_path))[0]

    skeleton, dist, segments, segment_types = extract_skeleton_and_segments(
        mask, min_length=args.min_length
    )
    branch = get_branch_points(skeleton)
    endpts = get_endpoints(skeleton)

    # Branch/endpoint coordinates for scatter
    br_r, br_c = np.where(branch)
    ep_r, ep_c = np.where(endpts)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(script_dir, "visual", f"{case_id}_segments.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(args.fig_width, args.fig_width))
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Segment Extraction Visualization (min_length={args.min_length})\nCase: {case_id}", fontsize=12, color="white")

    # 1. Original Mask (black bg, white vessels)
    ax = axes[0, 0]
    ax.set_facecolor("black")
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title("1. Original Mask", color="white")
    ax.axis("off")

    # 2. Skeleton (black bg, white lines)
    ax = axes[0, 1]
    ax.set_facecolor("black")
    skel_show = np.zeros((*mask.shape, 4))
    skel_show[:, :, :3] = 0.0  # black bg
    skel_show[skeleton > 0, :3] = 1.0  # white lines
    skel_show[:, :, 3] = 1.0
    ax.imshow(skel_show)
    ax.set_title("2. Skeleton (White)", color="white")
    ax.axis("off")

    # 3. Branch points & Endpoints (skeleton + yellow squares / cyan points)
    ax = axes[1, 0]
    ax.set_facecolor("black")
    skel_show = np.zeros((*mask.shape, 4))
    skel_show[:, :, :3] = 0.0
    skel_show[skeleton > 0, :3] = 1.0  # white skeleton
    skel_show[:, :, 3] = 1.0
    ax.imshow(skel_show)
    ax.scatter(br_c, br_r, c="gold", s=18, marker="s", label="Branch Points", zorder=3, edgecolors="white")
    ax.scatter(ep_c, ep_r, c="cyan", s=25, marker="o", label="Endpoints", zorder=3, edgecolors="white")
    leg = ax.legend(loc="upper right", fontsize=8, facecolor="black", edgecolor="white")
    for t in leg.get_texts():
        t.set_color("white")
    ax.set_title("3. Branch Points & Endpoints", color="white")
    ax.axis("off")

    # 4. All segments (extended palette)
    ax = axes[1, 1]
    ax.set_facecolor("black")
    seg_canvas = np.zeros((*mask.shape, 4))
    seg_canvas[:, :, :3] = 0.0  # black bg
    seg_canvas[:, :, 3] = 1.0
    colors = _segment_colors(len(segments))
    for seg, color in zip(segments, colors):
        rgb = matplotlib.colors.to_rgba(color)[:3]
        for ((r, c), _) in seg:
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                seg_canvas[r, c, :3] = rgb
                seg_canvas[r, c, 3] = 1.0
    ax.imshow(seg_canvas)
    ax.set_title(f"4. All Segments ({len(segments)} segments, min_length={args.min_length}) Using Extended Color Palette", color="white")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(
        out_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
