# 用 ROP 数据训练 Segmentation-Guided Diffusion

基于本仓库在 **ROP（早产儿视网膜病变）** 数据上重新训练分割引导扩散模型。

## 1. 仓库在做什么

- **`main.py`**：入口，支持 `train` / `eval` / `eval_many`。
- **`training.py`**：训练循环、ref 条件（同标签参考图）等。
- **`eval.py`**：推理时把分割 mask 拼到噪声上（`add_segmentations_to_noise`），支持 DDIM/DDPM pipeline。

支持的两种数据布局：

| 布局 | 说明 |
|------|------|
| **标准布局** | `img_dir`: `train/`, `val/`, `test/`；`seg_dir/all/`: `train/`, `val/`, `test/`，mask 与图像同名。 |
| **CROP 布局** | `img_dir` = `seg_dir` = 同一根目录，下含 `train_images/`, `val_images/`, `test_images/`，每类下再含 `*_images/` 与 `*_masks/`。 |

ROP 脚本默认用 **CROP 布局**（与现有 `run_train_crop_ref.sh` 一致）。

## 2. ROP 数据准备

### 2.1 目录结构（CROP 布局）

把 ROP 图像和 mask 整理成如下结构（与 CROP 相同）：

```
ROP_ROOT/
├── train_images/
│   ├── train_images/    # 训练图像 *.png
│   └── train_masks/     # 对应 mask，与图像同名
├── val_images/
│   ├── val_images/
│   └── val_masks/
└── test_images/
    ├── test_images/
    └── test_masks/
```

- **图像**：RGB，任意分辨率，训练时会 resize 到 `--img_size`（默认 256）。
- **Mask**：单通道/灰度，像素值为**整数类别**：`0` = 背景，`1` = 血管（与 nnUNet `Dataset008_ROP_CROP` 的 `labels` 一致）。文件名与对应图像**严格同名**。

### 2.2 可选：诊断标签（用于 ref 条件）

若要做“同标签参考图”条件生成（与 `run_train_crop_ref.sh` 类似），需要：

- 在 `LABEL_DIR` 下放置 CSV：`train_1.csv`、`valid.csv`、`test.csv`。
- 每表至少两列：`id_code`（与图像文件名不含扩展名一致）、`diagnosis`（整数标签）。

然后在 `run_train_rop.sh` 中设置 `LABEL_DIR` 并取消注释 `--use_ref_conditioning`、`--ref_downsize 64`、`--label_dir "$LABEL_DIR"`。

## 3. 训练命令

```bash
cd /media/yuganlab/blackstone/MICCAI2026/segmentation-guided-diffusion

# 指定你的 ROP 数据根目录
export ROP_ROOT=/path/to/your/ROP_dataset
# 可选：ref 条件用的标签目录
# export LABEL_DIR=/path/to/your/ROP_labels

bash run_train_rop.sh
```

- 输出目录：`ddpm-ROP-256-segguided`（若启用 ref 则为 `ddpm-ROP-256-segguided-ref`）。
- 模型保存在该目录下的 `unet/`（如 `diffusion_pytorch_model.safetensors` 或 `.bin`）。

## 4. 主要参数说明

| 参数 | 当前默认 | 说明 |
|------|----------|------|
| `--img_size` | 256 | 训练/推理时的图像边长 |
| `--num_img_channels` | 3 | RGB |
| `--num_segmentation_classes` | 2 | 0=背景, 1=血管（含背景） |
| `--model_type` | DDPM | 也可改为 DDIM |
| `--train_batch_size` | 16 | 按显存调整 |
| `--num_epochs` | 300 | 可自行修改 |
| `--use_crop_data` | 已开 | 使用 CROP 布局 |
| `--cuda_device` | 0 | 指定 GPU |

更多选项见：`python main.py --help`，以及 `training.py` 里 `TrainingConfig`。

## 5. 从现有 CROP/APTOS 到 ROP 的对应关系

- 现有 `run_train_crop_ref.sh`：`dataset=CROP`，`img_dir/seg_dir` 指向 CROP 根目录，`label_dir` 指向 aptos2019 CSV，启用 `--use_ref_conditioning`。
- ROP 脚本：`dataset=ROP`，`img_dir/seg_dir` 指向你的 ROP 根目录，分割类别数为 2；是否用 ref 条件由你是否提供 `LABEL_DIR` 并取消注释决定。

按上述准备好 ROP 的 CROP 布局（及可选标签）后，直接运行 `run_train_rop.sh` 即可开始训练。
