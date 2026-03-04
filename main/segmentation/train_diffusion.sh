#!/bin/bash

# Train Segmentation-Guided Diffusion (APTOS2019, 256x256)

set -e

eval "$(conda shell.bash hook)"
conda activate seg_diff

CUDA_DEVICE=2
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

DATASET_NAME="aptos2019"
IMAGE_SIZE=256
NUM_IMG_CHANNELS=3  # RGB
NUM_SEG_CLASSES=2   # background(0) + vessel(1)

IMG_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/APTOS2019_DIFFUSION/images"
MASK_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/datasets/APTOS2019_DIFFUSION/masks"

TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=8
NUM_EPOCHS=400
MODEL_TYPE="DDPM"

WORK_DIR="/media/yuganlab/blackstone/MICCAI2026/segmentation-guided-diffusion"
cd ${WORK_DIR}

echo "=========================================="
echo "Train Segmentation-Guided Diffusion"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Channels: ${NUM_IMG_CHANNELS} (RGB)"
echo "Seg classes: ${NUM_SEG_CLASSES}"
echo "GPU: CUDA:${CUDA_DEVICE}"
echo "Model: ${MODEL_TYPE}"
echo "Train batch: ${TRAIN_BATCH_SIZE}"
echo "Eval batch: ${EVAL_BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="
echo ""

if [ ! -d "${IMG_DIR}" ]; then
    echo "Error: image dir missing: ${IMG_DIR}"
    exit 1
fi

if [ ! -d "${MASK_DIR}" ]; then
    echo "Error: mask dir missing: ${MASK_DIR}"
    exit 1
fi

echo "Starting training..."
echo ""

python3 main.py \
    --mode train \
    --model_type ${MODEL_TYPE} \
    --img_size ${IMAGE_SIZE} \
    --num_img_channels ${NUM_IMG_CHANNELS} \
    --dataset ${DATASET_NAME} \
    --img_dir ${IMG_DIR} \
    --seg_dir ${MASK_DIR} \
    --segmentation_guided \
    --num_segmentation_classes ${NUM_SEG_CLASSES} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS}

echo ""
echo "=========================================="
echo "Training done."
echo "=========================================="
