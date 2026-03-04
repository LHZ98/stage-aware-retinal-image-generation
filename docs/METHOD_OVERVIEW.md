# Method Overview (Fig. 1)

This document is a text version of **Fig. 1. Overview of the proposed stage-aware retinal image generation framework**, and how it maps to this repository.

---

## 1. Geometry-Aware Vascular Transformation

**Input:** Fundus image.

1. **Segmentation**  
   The fundus image is passed through **nn-UNet** to obtain a **vessel mask** (binary or multi-class).

2. **Geometric feature extraction**  
   - **Skeletonize:** Reduce the vessel mask to a one-pixel-wide skeleton.  
   - **Branch/End points extract:** Identify branch and end points of the vascular network.

3. **Quantitative geometric measures**  
   From each vessel segment we compute:
   - **Segments (length)** \(L_i\):  
     \(L_i = \int_0^1 \|y'_i(s)\| \, ds\)
   - **Tortuosity** \(T_i\):  
     \(T_i = L_i / \|y_i(1) - y_i(0)\|\)
   - **Caliber** \(C_i\):  
     \(C_i = (1/L_i) \int_0^{L_i} r(x) \, dl\)

4. **Target-stage transformation**  
   Using these measures, we generate **stage-specific masks** (e.g. Stage 1: Mild, Stage 4: Severe) that reflect the vascular geometry of the target DR stage.

**Output:** Generated masks for the desired stages.

**Code:** `nnUNet/`, `main/vessel/` (skeleton, tortuosity, width, metrics), `main/mask_generation/` (stage-specific mask generation).

---

## 2. Structure & Appearance Conditioned Diffusion Training

**Inputs:**
- Fundus image (clean, \(X_0\))
- **Mask reference** \(M_r\) (vessel/structure mask)
- **Appearance reference** \(A_r\) (reference fundus image)

**Process:**
- Forward diffusion: \(X_0 \to X_1 \to \cdots \to X_T\) (noise level increases).
- At each step \(t\), a **conditional noise predictor** (in the figure: ViT; in this repo: UNet-based model) predicts the noise in \(X_t\) given the mask and appearance references:
  \[
  \epsilon_\theta(x_t, t \mid M_r, A_r)
  \]
- This model is trained so that the reverse (denoising) process can generate images that match \(M_r\) and the appearance of \(A_r\).

**Output:** Trained diffusion checkpoint for conditional generation.

**Code:** `segmentation-guided-diffusion/` (training and evaluation with mask and optional appearance conditioning).

---

## 3. Stage-Controlled Sampling

**Inputs:**
- **Generated masks** from Stage 1 (e.g. Stage 1: Mild, Stage 4: Severe)
- **Reference image** (appearance reference)

**Process:**
- The **DDPM** trained in Stage 2 is used to denoise, conditioned on the generated masks and the reference image.
- **ResNet-50** (DR stage classifier, 0–4) provides **stage guidance**: at each denoising step, the classifier gradient is used to steer the sample toward the target stage (classifier-guided sampling, as in Diffusion Visual Counterfactual Explanations).
- The combination of structure (masks), appearance (reference), and stage (classifier) yields stage-controlled samples.

**Output:** Generated fundus images with the desired structure, appearance, and stage.

**Code:** `segmentation-guided-diffusion/run_generate_from_masks.py` (mask + ref conditioning); classifier guidance uses the model in `classify/DR/` and follows the logic in [retinal_image_counterfactuals](https://github.com/berenslab/retinal_image_counterfactuals) (e.g. `cond_fn` in the diffusion sampling loop).

---

## Summary

| Stage | Name | Main output |
|-------|------|-------------|
| 1 | Geometry-aware vascular transformation | Stage-specific vessel masks |
| 2 | Structure & appearance conditioned diffusion training | Trained DDPM (conditioned on \(M_r\), \(A_r\)) |
| 3 | Stage-controlled sampling | Generated images (structure + appearance + stage-guided) |

For repository layout and run instructions, see the root **README.md**.
