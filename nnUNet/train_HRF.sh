#!/bin/bash

# nnUNet HRF数据集训练脚本
# 包含：数据转换 -> 实验规划 -> 模型训练

# ============================================
# 配置参数
# ============================================

# Conda环境名称（请根据实际情况修改）
CONDA_ENV="nnunet"  # 或者你的环境名称

# 数据集路径
IMAGES_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/HRF/images"
MASKS_DIR="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/HRF/manual1"

# 数据集ID和名称
DATASET_ID="001"
DATASET_NAME="HRF"

# nnUNet环境变量（请根据实际情况修改路径）
export nnUNet_raw="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_raw"
export nnUNet_preprocessed="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_preprocessed"
export nnUNet_results="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_results"

# GPU设置
CUDA_DEVICE=0  # 使用哪个GPU，可以改为0, 1, 2等
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
# 步骤1: 数据转换
# ============================================

echo ""
echo "=========================================="
echo "步骤1: 转换数据到nnUNet格式"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# 检查输入路径
if [ ! -d "${IMAGES_DIR}" ]; then
    echo "错误: 图像文件夹不存在: ${IMAGES_DIR}"
    exit 1
fi

if [ ! -d "${MASKS_DIR}" ]; then
    echo "错误: Mask文件夹不存在: ${MASKS_DIR}"
    exit 1
fi

# 创建输出目录
mkdir -p "${nnUNet_raw}"
mkdir -p "${nnUNet_preprocessed}"
mkdir -p "${nnUNet_results}"

# 运行转换脚本
python convert_retinal_fundus_to_nnunet.py \
    --images_dir "${IMAGES_DIR}" \
    --masks_dir "${MASKS_DIR}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${nnUNet_raw}"

if [ $? -ne 0 ]; then
    echo "错误: 数据转换失败！"
    exit 1
fi

echo "✅ 数据转换完成！"

# ============================================
# 步骤2: 实验规划和预处理
# ============================================

echo ""
echo "=========================================="
echo "步骤2: 实验规划和预处理"
echo "=========================================="

nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

if [ $? -ne 0 ]; then
    echo "错误: 实验规划失败！"
    exit 1
fi

echo "✅ 实验规划完成！"

# ============================================
# 步骤3: 训练模型（5折交叉验证）
# ============================================

echo ""
echo "=========================================="
echo "步骤3: 开始训练模型（${NUM_FOLDS}折交叉验证）"
echo "=========================================="

# 训练所有folds
for FOLD in $(seq 0 $((NUM_FOLDS-1))); do
    echo ""
    echo "----------------------------------------"
    echo "训练 Fold ${FOLD}/${NUM_FOLDS-1}"
    echo "----------------------------------------"
    
    if [ -n "${TRAINER}" ]; then
        nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLD} --npz -tr ${TRAINER}
    else
        nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLD} --npz
    fi
    
    if [ $? -ne 0 ]; then
        echo "警告: Fold ${FOLD} 训练失败，继续下一个fold..."
    else
        echo "✅ Fold ${FOLD} 训练完成！"
    fi
done

echo ""
echo "=========================================="
echo "所有Folds训练完成！"
echo "=========================================="

# ============================================
# 步骤4: 找到最佳配置
# ============================================

echo ""
echo "=========================================="
echo "步骤4: 分析最佳配置"
echo "=========================================="

nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIGURATION}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 最佳配置分析完成！"
    echo ""
    echo "查看结果:"
    echo "  - 推理指令: ${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/inference_instructions.txt"
    echo "  - 性能信息: ${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/inference_information.json"
else
    echo "警告: 最佳配置分析失败，但训练已完成。"
fi

echo ""
echo "=========================================="
echo "🎉 训练流程全部完成！"
echo "=========================================="
echo ""
echo "下一步: 使用 test_HRF.sh 进行预测"
echo ""

