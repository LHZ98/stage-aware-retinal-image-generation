#!/bin/bash

# nnUNet FIVE training (original data, 5-fold CV, fold 0 only). GPU: CUDA:0

CONDA_ENV="nnunet"

TRAIN_IMAGES_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/train/Original"
TRAIN_MASKS_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/train/Ground truth"
TEST_IMAGES_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/test/Original"
TEST_MASKS_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/test/Ground truth"

DATASET_ID="006"
DATASET_NAME="FIVE"

export nnUNet_raw="/media/yuganlab/blackstone/MICCAI2026/nnUNet_raw"
export nnUNet_preprocessed="/media/yuganlab/blackstone/MICCAI2026/nnUNet_preprocessed"
export nnUNet_results="/media/yuganlab/blackstone/MICCAI2026/nnUNet_results"

CUDA_DEVICE=0
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

CONFIGURATION="2d"
FOLD=0

TRAINER=""
export nnUNet_n_proc_DA=12
export nnUNet_compile="False"

echo "=========================================="
echo "Train: ${DATASET_NAME} (original)"
echo "GPU: CUDA:${CUDA_DEVICE}, Fold: ${FOLD}, Epochs: 1000"
echo "=========================================="

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV} 2>/dev/null || {
        echo "Warning: cannot activate conda env ${CONDA_ENV}"
    }
fi

python -c "import nnunetv2; print('nnUNet OK')" 2>/dev/null || {
    echo "Error: nnUNet not installed"
    exit 1
}

echo ""
echo "=========================================="
echo "Step 1: Convert data to nnUNet format (original)"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

if [ ! -d "${TRAIN_IMAGES_DIR}" ]; then
    echo "Error: train images dir missing: ${TRAIN_IMAGES_DIR}"
    exit 1
fi

if [ ! -d "${TRAIN_MASKS_DIR}" ]; then
    echo "Error: train masks dir missing: ${TRAIN_MASKS_DIR}"
    exit 1
fi

mkdir -p "${nnUNet_raw}"
mkdir -p "${nnUNet_preprocessed}"
mkdir -p "${nnUNet_results}"

python /media/yuganlab/blackstone/MICCAI2026/nnUNet/convert_retinal_fundus_to_nnunet.py \
    --images_dir "${TRAIN_IMAGES_DIR}" \
    --masks_dir "${TRAIN_MASKS_DIR}" \
    --output_dir "${nnUNet_raw}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_name "${DATASET_NAME}" \
    --file_ending ".png" \
    --target_size 512

if [ $? -ne 0 ]; then
    echo "Error: data conversion failed"
    exit 1
fi

echo "Data conversion done."
echo ""

echo "=========================================="
echo "Step 2: Plan and preprocess"
echo "=========================================="

nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    --verify_dataset_integrity

if [ $? -ne 0 ]; then
    echo "Error: planning failed"
    exit 1
fi

echo "Planning done."
echo ""

echo "=========================================="
echo "Step 3: Train (Fold ${FOLD}, 5-fold CV)"
echo "=========================================="

TRAIN_CMD="nnUNetv2_train"

if [ -n "${TRAINER}" ]; then
    TRAIN_CMD="${TRAIN_CMD} -tr ${TRAINER}"
    echo "Trainer: ${TRAINER}"
else
    echo "Default Trainer (1000 epochs)"
fi

echo ""
echo "----------------------------------------"
echo "Fold ${FOLD}, GPU: CUDA:${CUDA_DEVICE}"
echo "Epochs: 1000, checkpoint every 50"
echo "----------------------------------------"

${TRAIN_CMD} \
    "${DATASET_ID}" \
    "${CONFIGURATION}" \
    "${FOLD}" \
    --npz

if [ $? -ne 0 ]; then
    echo "Error: training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training done."
echo "=========================================="
echo ""
echo "Dataset: Dataset${DATASET_ID}_${DATASET_NAME}"
echo "Config: ${CONFIGURATION}, Fold: ${FOLD}, GPU: CUDA:${CUDA_DEVICE}"
echo "Results: ${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/"
echo "Checkpoint: .../nnUNetTrainer__nnUNetPlans__2d/fold_${FOLD}/"
echo ""
