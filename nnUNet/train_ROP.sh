#!/bin/bash

# nnUNet ROP数据集训练脚本
# 包含：数据转换 -> 实验规划 -> 模型训练

# ============================================
# 配置参数
# ============================================

# Conda环境名称（请根据实际情况修改）
CONDA_ENV="nnunet"  # 或者你的环境名称

# 数据集路径（ROP数据集图像和mask在同一目录，通过文件名匹配）
# 图像文件：*.jpg
# Mask文件：*_mask.jpg
ROP_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/ROP"

# 创建临时目录用于组织数据
TEMP_IMAGES_DIR="${ROP_DIR}/Original"
TEMP_MASKS_DIR="${ROP_DIR}/Ground_truth"

# 数据集ID和名称
DATASET_ID="005"
DATASET_NAME="ROP"

# nnUNet环境变量（请根据实际情况修改路径）
export nnUNet_raw="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_raw"
export nnUNet_preprocessed="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_preprocessed"
export nnUNet_results="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_results"

# GPU设置
CUDA_DEVICE=1  # 使用哪个GPU，可以改为0, 1, 2等
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# 训练配置
CONFIGURATION="2d"  # 对于2D视网膜图像，使用2d配置
NUM_FOLDS=5  # 5折交叉验证

# Trainer选择（可选：使用增强版Trainer处理domain shift）
# 默认使用标准Trainer，如需使用增强版，取消下面一行的注释
# TRAINER="nnUNetTrainerEnhancedLightness"  # 增强版，更强的lightness扰动
TRAINER=""  # 空值表示使用默认Trainer

# 数据增强进程数（根据CPU核心数调整）
export nnUNet_n_proc_DA=12  # RTX 3090推荐12，RTX 4090推荐16-18

# 禁用torch.compile以避免兼容性问题（如果遇到训练错误，可以尝试禁用）
export nnUNet_compile="False"

# ============================================
# 激活环境
# ============================================

echo "=========================================="
echo "激活Conda环境: ${CONDA_ENV}"
echo "=========================================="

# 尝试激活conda环境
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV} 2>/dev/null || {
        echo "警告: 无法激活conda环境 ${CONDA_ENV}，尝试使用系统Python"
    }
else
    echo "警告: 未找到conda，使用系统Python"
fi

# 检查nnUNet是否安装
python -c "import nnunetv2; print('nnUNet已安装')" 2>/dev/null || {
    echo "错误: nnUNet未安装！请先安装nnUNet。"
    exit 1
}

# ============================================
# 步骤0: 组织ROP数据（图像和mask分离）
# ============================================

echo ""
echo "=========================================="
echo "步骤0: 组织ROP数据集结构"
echo "=========================================="

# 创建临时目录
mkdir -p "${TEMP_IMAGES_DIR}"
mkdir -p "${TEMP_MASKS_DIR}"

# 统计文件数量
IMAGE_COUNT=$(ls "${ROP_DIR}"/*.jpg 2>/dev/null | grep -v "_mask" | wc -l)
MASK_COUNT=$(ls "${ROP_DIR}"/*_mask.jpg 2>/dev/null | wc -l)

echo "发现 ${IMAGE_COUNT} 个图像文件"
echo "发现 ${MASK_COUNT} 个mask文件"

if [ ${IMAGE_COUNT} -eq 0 ] || [ ${MASK_COUNT} -eq 0 ]; then
    echo "错误: 未找到足够的图像或mask文件！"
    exit 1
fi

# 创建符号链接或复制文件到临时目录
echo "组织文件结构..."
for img_file in "${ROP_DIR}"/*.jpg; do
    if [[ ! "${img_file}" =~ _mask\.jpg$ ]]; then
        # 这是图像文件
        filename=$(basename "${img_file}")
        # 创建符号链接（如果失败则复制）
        ln -sf "${img_file}" "${TEMP_IMAGES_DIR}/${filename}" 2>/dev/null || \
        cp "${img_file}" "${TEMP_IMAGES_DIR}/${filename}"
        
        # 查找对应的mask文件
        mask_file="${ROP_DIR}/${filename%.jpg}_mask.jpg"
        if [ -f "${mask_file}" ]; then
            mask_filename=$(basename "${mask_file}")
            # 移除_mask后缀，保持文件名一致
            base_name="${filename%.jpg}"
            ln -sf "${mask_file}" "${TEMP_MASKS_DIR}/${base_name}.jpg" 2>/dev/null || \
            cp "${mask_file}" "${TEMP_MASKS_DIR}/${base_name}.jpg"
        fi
    fi
done

ORGANIZED_IMAGES=$(ls "${TEMP_IMAGES_DIR}"/*.jpg 2>/dev/null | wc -l)
ORGANIZED_MASKS=$(ls "${TEMP_MASKS_DIR}"/*.jpg 2>/dev/null | wc -l)

echo "已组织 ${ORGANIZED_IMAGES} 个图像文件"
echo "已组织 ${ORGANIZED_MASKS} 个mask文件"

if [ ${ORGANIZED_IMAGES} -eq 0 ] || [ ${ORGANIZED_MASKS} -eq 0 ]; then
    echo "错误: 文件组织失败！"
    exit 1
fi

echo "✅ 数据组织完成！"
echo ""

# ============================================
# 步骤1: 数据转换
# ============================================

echo "=========================================="
echo "步骤1: 转换数据到nnUNet格式"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# 检查输入路径
if [ ! -d "${TEMP_IMAGES_DIR}" ]; then
    echo "错误: 图像文件夹不存在: ${TEMP_IMAGES_DIR}"
    exit 1
fi

if [ ! -d "${TEMP_MASKS_DIR}" ]; then
    echo "错误: Mask文件夹不存在: ${TEMP_MASKS_DIR}"
    exit 1
fi

# 创建输出目录
mkdir -p "${nnUNet_raw}"
mkdir -p "${nnUNet_preprocessed}"
mkdir -p "${nnUNet_results}"

# 运行转换脚本
python convert_retinal_fundus_to_nnunet.py \
    --images_dir "${TEMP_IMAGES_DIR}" \
    --masks_dir "${TEMP_MASKS_DIR}" \
    --output_dir "${nnUNet_raw}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_name "${DATASET_NAME}" \
    --file_ending ".png" \
    --target_size 512

if [ $? -ne 0 ]; then
    echo "错误: 数据转换失败！"
    exit 1
fi

echo "数据转换完成！"
echo ""

# ============================================
# 步骤2: 实验规划和预处理
# ============================================

echo "=========================================="
echo "步骤2: 实验规划和预处理"
echo "=========================================="

nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    --verify_dataset_integrity

if [ $? -ne 0 ]; then
    echo "错误: 实验规划失败！"
    exit 1
fi

echo "实验规划完成！"
echo ""

# ============================================
# 步骤3: 模型训练（5折交叉验证）
# ============================================

echo "=========================================="
echo "步骤3: 开始5折交叉验证训练"
echo "=========================================="

# 构建训练命令
TRAIN_CMD="nnUNetv2_train"

# 如果指定了Trainer，添加参数
if [ -n "${TRAINER}" ]; then
    TRAIN_CMD="${TRAIN_CMD} -tr ${TRAINER}"
    echo "使用Trainer: ${TRAINER}"
else
    echo "使用默认Trainer"
fi

# 训练所有fold
for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
    echo ""
    echo "----------------------------------------"
    echo "训练 Fold ${FOLD}/${NUM_FOLDS}"
    echo "----------------------------------------"
    
    ${TRAIN_CMD} \
        "${DATASET_ID}" \
        "${CONFIGURATION}" \
        "${FOLD}" \
        --npz
    
    if [ $? -ne 0 ]; then
        echo "警告: Fold ${FOLD} 训练失败，继续下一个fold..."
    else
        echo "Fold ${FOLD} 训练完成！"
    fi
done

echo ""
echo "=========================================="
echo "所有Fold训练完成！"
echo "=========================================="

# ============================================
# 步骤4: 选择最佳配置（可选）
# ============================================

echo ""
echo "=========================================="
echo "步骤4: 查找最佳配置"
echo "=========================================="

nnUNetv2_find_best_configuration \
    -d "${DATASET_ID}" \
    -c "${CONFIGURATION}" \
    -tr "${TRAINER:-nnUNetTrainer}"

if [ $? -ne 0 ]; then
    echo "警告: 查找最佳配置失败，但训练已完成"
fi

echo ""
echo "=========================================="
echo "训练流程全部完成！"
echo "=========================================="
echo ""
echo "数据集ID: Dataset${DATASET_ID}_${DATASET_NAME}"
echo "配置: ${CONFIGURATION}"
echo "结果保存在: ${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/"
echo ""
echo "注意: 临时组织的数据在:"
echo "  - ${TEMP_IMAGES_DIR}"
echo "  - ${TEMP_MASKS_DIR}"
echo "  如需清理，可以删除这些目录"
echo ""

