#!/usr/bin/env bash
# 使用 HVDROPDB ROP 数据训练 segmentation-guided diffusion
# 数据布局（flat）：
#   IMG_DIR/input_nnunet/*.png           (图像，如 Neo_Normal__10_0000.png)
#   IMG_DIR/pred_masks_by_class_denoised/<Prefix>/<num>.png  (mask 与图像按文件名对应)
# 训练/验证 90/10 随机划分（固定种子）

set -e
cd "$(dirname "$0")"

# HVDROPDB ROP 数据根目录（下含 input_nnunet 与 pred_masks_by_class_denoised）
ROP_ROOT="${ROP_ROOT:-/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/HVDROPDB_CROP_nnunet_pred_Dataset008_fold0_best}"

IMG_DIR="$ROP_ROOT"
SEG_DIR="$ROP_ROOT"

# 2 类分割：0=背景, 1=血管（含背景）
NUM_SEG_CLASSES=2

python main.py \
  --mode train \
  --img_size 256 \
  --num_img_channels 3 \
  --dataset ROP \
  --img_dir "$IMG_DIR" \
  --seg_dir "$SEG_DIR" \
  --model_type DDPM \
  --segmentation_guided \
  --segmentation_channel_mode single \
  --num_segmentation_classes $NUM_SEG_CLASSES \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --num_epochs 300 \
  --use_flat_data \
  --img_subdir input_nnunet \
  --mask_subdir pred_masks_by_class_denoised \
  --cuda_device 0
