# HRF Dataset: nnUNet Training and Prediction

This guide describes how to train an nnUNet model on the HRF (High-Resolution Fundus) retinal dataset and run prediction.

## Files

- `train_HRF.sh`: Full pipeline (convert → plan → train)
- `test_HRF.sh`: Prediction script
- `nnunetv2/.../nnUNetTrainerEnhancedLightness.py`: Trainer with stronger lightness augmentation

## Quick Start

### 1. Environment variables

Set nnUNet paths in `train_HRF.sh` and `test_HRF.sh` (or in `.bashrc`):

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### 2. Conda environment

Set `CONDA_ENV="nnunet"` (or your env name) in both scripts.

### 3. Train

```bash
cd /path/to/nnUNet
bash train_HRF.sh
```

The script will: convert data, plan and preprocess, train with 5-fold CV, and run best-configuration analysis.

### 4. Predict

```bash
bash test_HRF.sh /path/to/test/images /path/to/output/predictions
```

## Enhanced Trainer (Domain Shift)

For stronger lightness augmentation, use `nnUNetTrainerEnhancedLightness`:

In `train_HRF.sh`:
```bash
nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLD} --npz -tr nnUNetTrainerEnhancedLightness
```

Or from the command line:
```bash
conda activate nnunet
# set nnUNet_* env vars
nnUNetv2_train 001 2d 0 --npz -tr nnUNetTrainerEnhancedLightness
```

**Enhanced Trainer** uses stronger brightness, contrast, and gamma augmentation to improve robustness across imaging conditions.

## Outputs

- **Training**: `nnUNet_results/Dataset001_HRF/nnUNetTrainer__nnUNetPlans__2d/fold_*`
- **Prediction**: PNG masks (integer labels); optional NPZ probability maps with `--save_probabilities`

## FAQ

**Q: Training progress?**  
A: See `progress.png` in each fold directory.

**Q: Resume training?**  
A: Add `--c`: `nnUNetv2_train 001 2d 0 --npz --c`

**Q: Multi-GPU?**  
A: Use `-num_gpus X` or run folds on different GPUs with `CUDA_VISIBLE_DEVICES`.

**Q: Tune augmentation?**  
A: Edit `nnUNetTrainerEnhancedLightness.py` (ranges and probabilities).
