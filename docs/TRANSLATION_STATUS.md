# Translation status (Chinese → English)

Repository content is being converted to English (READMEs, code, comments).

## Done (full or major)

- **Root:** `README.md`, `docs/METHOD_OVERVIEW.md` — English
- **classify/DR:** `train.py`, `config.py`, `dataset.py`, `metrics.py` — English
- **main/vessel:** `README.md`, `tortuosity_control.py`, `path_utils.py`, `experiment.py`, `neovascularization.py`, `create_demo_mask.py`, `run_batch.py`, `test_two_examples.py` — English
- **main/mask_generation:** All listed scripts — docstrings and user-facing strings in English
- **main/segmentation:** All `*.py` and `*.sh` — English (`assess_diffusion_data.py`, `prepare_diffusion_training.py`, `crop_aptos2019.py`, `train_*.sh`, `train_diffusion.sh`, `train_both.sh`, `check_diffusion_samples.sh`, `setup_seg_diff_env.sh`)
- **nnUNet:** `CONVERT_README.md` — English; `convert_mask_to_0_255.py`, `convert_retinal_fundus_to_nnunet.py` — user-facing strings and argparse in English; `visualize_predictions.py`, `crop_train_samples.py`, `nnUNetTrainerEnhancedLightness.py` — top-level docstrings in English
- **segmentation-guided-diffusion:** `main.py` — English

## Still containing Chinese (optional)

- **nnUNet:** `README_HRF.md`, inline comments in `convert_retinal_fundus_to_nnunet.py` / `crop_train_samples.py` / `visualize_predictions.py` / `nnUNetTrainerEnhancedLightness.py`, `preprocess/download_ROP_dataset.py`, `preprocess/crop_fundus_images.py`, `convert_example.sh`, `train_*.sh`, `test_*.sh`
- **segmentation-guided-diffusion:** `README_ROP.md`, `run_train_rop.sh`

To find remaining Chinese in a directory:

```bash
grep -r -l -P '[\x{4e00}-\x{9fff}]' --include='*.py' --include='*.md' --include='*.sh' .
```
