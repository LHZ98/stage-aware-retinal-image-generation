# Geometry-Aware Stage-Conditioned Vascular Transformation

This directory implements skeleton- and radius-based geometric transformation of vessel masks for controllable adjustment of:

- **Tortuosity**
- **Caliber**
- Optional **neovascularization (neo)**

Features:

- **Topology-friendly:** segment anchors (endpoints/branch points) are fixed by default
- **Interpretable:** parameters directly control frequency, amplitude, center protection, main-branch gain
- **Reproducible:** unified CLI entry with fixed random seed support

---

## 1. Pipeline

1) **Skeleton and segmentation** (`skeleton_utils.py`)  
   Binary mask → skeleton → segments (centerline points + local radius), labeled as `endpoint` / `main`.

2) **Path-level sinusoidal bending** (`path_utils.py` + `tortuosity_control.py`)  
   Continuous perturbation along endpoint→endpoint paths:
   - `delta = A_seg * sin(phi(s)) * n(s)`
   - `phi(s) = 2*pi*K*(s/L_path) + psi(s)`
   - `K` is `--sine_k_cycles` (smaller = lower frequency, sparser bends)
   - `A_seg` can be set by endpoint/main weights, main length gain, and random scale

3) **Optional center protection (recommended)**  
   Displacement decays toward the center. Center is computed as **radius-weighted density peak** (thick trunks weighted more), not geometric centroid.

4) **Optional neovascularization** (`neovascularization.py`)  
   Segment-level probability trigger, then endpoint vs mid-segment ratio for growth placement.

5) **Pre-reconstruction centerline smoothing (optional)**  
   Smooth interior points per segment, anchors unchanged, to reduce sharp kinks.

6) **Caliber control + capsule reconstruction** (`width_control.py`)  
   Radius scaling → Murray constraint → capsule reconstruction (point disks + neighbor capsules).

---

## 2. Entry and three run modes

Unified entry:

```bash
python -m main.vessel.experiment --mask <mask_path> --out_dir <output_dir>
```

### Mode A: `sine_only_visible` (most common)

Fixed amplitude for visual inspection only; no VTI bisection.

```bash
python -m main.vessel.experiment \
  --mask "datasets/masks/test/Ground truth/100_D.png" \
  --out_dir "main/vessel/test_output/demo_sine_visible" \
  --sine_only_visible --sine_along_paths \
  --sine_only_amplitudes 6 \
  --sine_k_cycles 0.8 \
  --sine_alpha_endpoint 1.0 \
  --sine_beta_main 0.8 \
  --sine_main_length_gain 0.5 \
  --sine_center_protect_radius 110 \
  --sine_center_protect_strength 0.8 \
  --sine_smooth_centerline --sine_smooth_window 5 --sine_smooth_iterations 1 \
  --add_neovascularization --neovascularization_prob 0.0 \
  --random_seed 42
```

### Mode B: `sine_path_vti` (target VTI → amplitude via bisection)

For each target VTI, bisect amplitude A so that achieved VTI approximates target.

```bash
python -m main.vessel.experiment \
  --mask "datasets/masks/test/Ground truth/100_D.png" \
  --out_dir "main/vessel/test_output/demo_path_vti" \
  --sine_path_vti \
  --vti_targets 1.0 1.1 1.2 \
  --width_scales 1.0 \
  --sine_k_cycles 0.8 \
  --sine_alpha_endpoint 1.0 \
  --sine_beta_main 0.8 \
  --sine_main_length_gain 0.5 \
  --sine_center_protect_radius 110 \
  --sine_center_protect_strength 0.8 \
  --sine_smooth_centerline \
  --random_seed 42
```

### Mode C: Default grid (legacy)

Without `--sine_only_visible` or `--sine_path_vti`, runs `run_experiment_grid` over a `vti_targets x width_scales` grid. Optional fixed amplitude via `--use_fixed_amplitude` / `--fixed_amplitude`.

---

## 3. CLI parameters (by group)

### 3.1 Basic I/O

- `--mask` (required): input binary vessel mask path (`.npy` or image)
- `--out_dir`: output directory

### 3.2 Grid (default / path_vti)

- `--vti_targets`: target VTI multipliers (default: `1.0 1.05 1.10`)
- `--width_scales`: caliber scale factors (default: `0.8 1.0 1.2`)

### 3.3 Neovascularization

- `--add_neovascularization`: enable neo generation
- `--neovascularization_prob`: segment-level trigger probability (default `0.1`). When triggered, 0.35 endpoint / 0.65 mid-segment.
- `--neovascularization_max_length`: max neo length in pixels

### 3.4 Randomness

- `--random_seed`: seed for tortuosity and neo (default `42`)

### 3.5 Fixed amplitude (default grid)

- `--use_fixed_amplitude`: skip VTI bisection, use fixed amplitude
- `--fixed_amplitude`: amplitude in pixels

### 3.6 Mode switches

- `--sine_only_visible`: fixed-amplitude “visible only” mode
- `--sine_only_amplitudes`: amplitude list in that mode (default `6 10 14`)
- `--sine_along_paths`: enable path-continuous sine in sine_only_visible
- `--sine_path_vti`: path sine + VTI closed-loop mode

### 3.7 Path sine shape

- `--sine_k_cycles`: frequency (waves per path), default `1.0`. Lower = sparser bends.
- `--sine_alpha_endpoint`: endpoint segment amplitude factor (default `0.9`)
- `--sine_beta_main`: main segment amplitude factor (default `1.0`)
- `--sine_main_length_gain`: main length gain (default `0.0`). Extra scale: `A_main *= (1 + gain * L_seg / L_path)`.

### 3.8 Center protection

- `--sine_center_protect_radius`: radius in pixels; >0 to enable
- `--sine_center_protect_strength`: strength in [0,1]. Center uses radius-weighted density peak.

### 3.9 Centerline smoothing

- `--sine_smooth_centerline`: enable pre-reconstruction smoothing
- `--sine_smooth_window`: window size (odd, default `5`)
- `--sine_smooth_iterations`: iterations (default `1`); `2` for smoother, less detail

---

## 4. Tuning tips

- **Fewer, sparser bends:** decrease `--sine_k_cycles` (e.g. 0.8 → 0.5 → 0.35).
- **Stronger main-branch bend:** increase `--sine_beta_main` or `--sine_main_length_gain`.
- **Less center distortion:** use `--sine_center_protect_radius 110 --sine_center_protect_strength 0.8`.
- **Smoother look:** `--sine_smooth_centerline`, then `--sine_smooth_window 5 --sine_smooth_iterations 1` (or 2).
- **No neo:** `--neovascularization_prob 0.0` or omit `--add_neovascularization`.

---

## 5. Input and output

**Input:** binary mask (`.npy` or image; thresholded at 0.5).

**Output (under `--out_dir`):** `mask_*.npy`, `metrics_*.json` (VTI, caliber, fractal dimension, and `amplitude_used` in path_vti mode), `viz_*.png`, `summary_table.csv` / `summary_table.txt`.

---

## 6. Environment

From repo root:

```bash
pip install -r main/vessel/requirements.txt
```

Or at least: `numpy`, `scipy`, `scikit-image`, `Pillow`, `matplotlib`.

---

## 7. Recommended config

Stable on several test masks with center protection and smoothing:

```bash
python -m main.vessel.experiment \
  --mask "datasets/masks/test/Ground truth/100_D.png" \
  --out_dir "main/vessel/test_output/recommended_v1" \
  --sine_only_visible --sine_along_paths \
  --sine_only_amplitudes 6 \
  --sine_k_cycles 0.8 \
  --sine_alpha_endpoint 1.0 \
  --sine_beta_main 0.8 \
  --sine_main_length_gain 0.5 \
  --sine_center_protect_radius 110 \
  --sine_center_protect_strength 0.8 \
  --sine_smooth_centerline --sine_smooth_window 5 --sine_smooth_iterations 1 \
  --add_neovascularization --neovascularization_prob 0.0 \
  --random_seed 42
```
