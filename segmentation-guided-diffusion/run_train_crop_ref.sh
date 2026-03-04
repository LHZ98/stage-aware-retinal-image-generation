#!/usr/bin/env bash
# Train Seg-Diff with ref conditioning on CROP (train_images + test_images + val_images),
# labels from aptos2019, ref = same-label within-batch, downscale 64 then upsample.
# Usage: adjust IMG_DIR, SEG_DIR, LABEL_DIR to your paths, then run.

set -e
cd "$(dirname "$0")"

IMG_DIR="${IMG_DIR:-/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP}"
SEG_DIR="${SEG_DIR:-/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/CROP}"  # supports CROP/<split>/<split>_masks layout
LABEL_DIR="${LABEL_DIR:-/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/aptos2019}"

python main.py \
  --mode train \
  --img_size 256 \
  --num_img_channels 3 \
  --dataset CROP \
  --img_dir "$IMG_DIR" \
  --seg_dir "$SEG_DIR" \
  --model_type DDPM \
  --segmentation_guided \
  --segmentation_channel_mode single \
  --num_segmentation_classes 2 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --num_epochs 300 \
  --use_ref_conditioning \
  --ref_downsize 64 \
  --use_crop_data \
  --label_dir "$LABEL_DIR" \
  --cuda_device 2
