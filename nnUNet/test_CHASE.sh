#!/bin/bash

# nnUNet CHASE_DB1数据集预测脚本
# 用于对测试图像进行分割预测

# ============================================
# 配置参数
# ============================================

# Conda环境名称（请根据实际情况修改）
CONDA_ENV="nnunet"  # 或者你的环境名称

# 数据集ID和配置
DATASET_ID="003"
DATASET_NAME="CHASE_DB1"
CONFIGURATION="2d"  # 使用训练时使用的配置

# nnUNet环境变量（必须与训练时一致）
export nnUNet_raw="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_raw"
export nnUNet_preprocessed="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_preprocessed"
export nnUNet_results="/home/yuganlab/Desktop/Haozhe/MICCAI2026/nnUNet_results"

# 输入和输出路径（可以从命令行参数获取）
INPUT_DIR="${1:-/path/to/test/images}"  # 从命令行参数获取，或使用默认路径
OUTPUT_DIR="${2:-./predictions/CHASE}"    # 预测结果输出路径

# 指定要使用的fold（可选，默认使用所有已训练的folds）
# 如果只想用单个fold，设置为 "0", "1", "2" 等
# 如果设置为 "all"，会使用所有已训练的folds进行集成
# 如果设置为空字符串 ""，会自动检测所有可用的folds
SPECIFIC_FOLD="${3:-}"  # 第三个参数：指定fold，如 "0" 表示只用fold_0

# GPU设置
CUDA_DEVICE=0  # 使用哪个GPU
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}

# 是否保存概率图（用于集成）
# 设置为空字符串则不保存npz和pkl文件，只保存PNG结果
SAVE_PROBABILITIES=""  # 不保存概率图，只保存PNG mask

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
# 检查输入路径
# ============================================

if [ ! -d "${INPUT_DIR}" ]; then
    echo ""
    echo "错误: 输入文件夹不存在: ${INPUT_DIR}"
    echo ""
    echo "使用方法:"
    echo "  bash test_CHASE.sh <输入图像文件夹> [输出文件夹] [fold编号]"
    echo ""
    echo "参数说明:"
    echo "  输入图像文件夹: 包含测试图像的文件夹（必需）"
    echo "  输出文件夹: 预测结果保存路径（可选，默认: ./predictions/CHASE）"
    echo "  fold编号: 指定使用的fold（可选，默认: 自动检测）"
    echo "            - 指定 '0' 表示只用 fold_0"
    echo "            - 指定 'all' 表示使用所有folds集成"
    echo "            - 不指定则自动使用所有可用的folds"
    echo ""
    echo "示例:"
    echo "  bash test_CHASE.sh /path/to/test/images ./predictions"
    echo "  bash test_CHASE.sh /path/to/test/images ./predictions 0    # 只用fold_0"
    echo "  bash test_CHASE.sh /path/to/test/images ./predictions all  # 使用所有folds"
    echo ""
    exit 1
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "预测配置"
echo "=========================================="
echo "数据集ID: ${DATASET_ID}"
echo "数据集名称: ${DATASET_NAME}"
echo "配置: ${CONFIGURATION}"
echo "输入文件夹: ${INPUT_DIR}"
echo "输出文件夹: ${OUTPUT_DIR}"
echo "GPU: ${CUDA_DEVICE}"
echo ""

# ============================================
# 检查模型是否存在
# ============================================

MODEL_DIR="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__${CONFIGURATION}"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "错误: 模型文件夹不存在: ${MODEL_DIR}"
    echo "请先运行 train_CHASE.sh 训练模型！"
    exit 1
fi

# 检查是否有训练好的folds
AVAILABLE_FOLDS=()
for FOLD in 0 1 2 3 4; do
    if [ -f "${MODEL_DIR}/fold_${FOLD}/checkpoint_final.pth" ]; then
        AVAILABLE_FOLDS+=(${FOLD})
    fi
done

if [ ${#AVAILABLE_FOLDS[@]} -eq 0 ]; then
    echo "错误: 未找到训练好的模型checkpoint！"
    echo "请先运行 train_CHASE.sh 训练模型！"
    exit 1
fi

echo "找到 ${#AVAILABLE_FOLDS[@]} 个训练好的folds: ${AVAILABLE_FOLDS[@]}"

# 确定要使用的fold
if [ -n "${SPECIFIC_FOLD}" ]; then
    # 用户指定了fold
    if [ "${SPECIFIC_FOLD}" = "all" ]; then
        USE_FOLD="all"
        echo "将使用所有可用的folds进行集成预测"
    else
        # 检查指定的fold是否存在
        if [[ " ${AVAILABLE_FOLDS[@]} " =~ " ${SPECIFIC_FOLD} " ]]; then
            USE_FOLD="${SPECIFIC_FOLD}"
            echo "将使用 fold_${USE_FOLD} 进行预测"
        else
            echo "错误: fold_${SPECIFIC_FOLD} 不存在或未训练完成！"
            echo "可用的folds: ${AVAILABLE_FOLDS[@]}"
            exit 1
        fi
    fi
else
    # 自动使用所有可用的folds
    if [ ${#AVAILABLE_FOLDS[@]} -eq 5 ]; then
        USE_FOLD="all"
        echo "将使用所有5个folds进行集成预测"
    else
        # 如果只有部分folds，使用第一个可用的
        USE_FOLD="${AVAILABLE_FOLDS[0]}"
        echo "注意: 只有部分folds训练完成，将使用 fold_${USE_FOLD} 进行预测"
        echo "可用的folds: ${AVAILABLE_FOLDS[@]}"
    fi
fi
echo ""

# ============================================
# 预处理输入图像（转换为PNG并添加_0000后缀）
# ============================================

echo "=========================================="
echo "预处理输入图像..."
echo "=========================================="

# 创建临时文件夹用于存储转换后的图像
TEMP_INPUT_DIR="${OUTPUT_DIR}/temp_input_images"
mkdir -p "${TEMP_INPUT_DIR}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 转换图像格式（如果需要）
python3 << EOF
import os
import glob
from PIL import Image
from pathlib import Path

input_dir = "${INPUT_DIR}"
output_dir = "${TEMP_INPUT_DIR}"

# 支持的输入格式
extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.gif']
input_files = []
for ext in extensions:
    input_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
    input_files.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))

print(f'找到 {len(input_files)} 个图像文件')

for img_file in input_files:
    try:
        # 获取case identifier
        case_id = Path(img_file).stem
        
        # nnUNet格式：{case_id}_0000.png
        output_file = os.path.join(output_dir, f'{case_id}_0000.png')
        
        # 读取并转换
        img = Image.open(img_file)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # 保存为PNG
        img.save(output_file, 'PNG')
    except Exception as e:
        print(f'警告: 转换 {os.path.basename(img_file)} 失败: {e}')

print(f'转换完成，输出到: {output_dir}')
EOF

if [ $? -ne 0 ]; then
    echo "警告: 图像预处理可能有问题，继续尝试预测..."
fi

# 使用转换后的图像进行预测
ACTUAL_INPUT_DIR="${TEMP_INPUT_DIR}"

# ============================================
# 运行预测
# ============================================

echo ""
echo "=========================================="
echo "开始预测..."
echo "=========================================="
echo ""

# 使用nnUNetv2_predict进行预测
if [ "${USE_FOLD}" = "all" ]; then
    # 使用所有folds进行集成预测
    nnUNetv2_predict \
        -i "${ACTUAL_INPUT_DIR}" \
        -o "${OUTPUT_DIR}" \
        -d ${DATASET_ID} \
        -c ${CONFIGURATION} \
        ${SAVE_PROBABILITIES}
else
    # 使用单个fold进行预测
    nnUNetv2_predict \
        -i "${ACTUAL_INPUT_DIR}" \
        -o "${OUTPUT_DIR}" \
        -d ${DATASET_ID} \
        -c ${CONFIGURATION} \
        -f ${USE_FOLD} \
        ${SAVE_PROBABILITIES}
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 预测完成！"
    echo "=========================================="
    echo ""
    
    # ============================================
    # 自动转换mask为0/255格式
    # ============================================
    echo "正在将mask转换为0/255格式..."
    python "${SCRIPT_DIR}/convert_mask_to_0_255.py" \
        --input_dir "${OUTPUT_DIR}" \
        --inplace
    
    if [ $? -eq 0 ]; then
        echo "✅ Mask转换完成（0/1 -> 0/255）"
    else
        echo "警告: Mask转换失败，但预测已完成"
    fi
    
    # 清理临时文件夹
    rm -rf "${TEMP_INPUT_DIR}"
    
    echo ""
    echo "预测结果保存在: ${OUTPUT_DIR}"
    echo ""
    echo "结果文件说明:"
    echo "  - *.png: 分割mask（binary: 0和255）"
    echo ""
else
    echo ""
    echo "错误: 预测失败！"
    # 清理临时文件夹
    rm -rf "${TEMP_INPUT_DIR}"
    exit 1
fi

# ============================================
# 可选: 应用后处理
# ============================================

# 如果训练时找到了最佳后处理策略，可以应用它
POSTPROCESSING_FILE="${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/postprocessing.pkl"

if [ -f "${POSTPROCESSING_FILE}" ]; then
    echo ""
    echo "=========================================="
    echo "应用后处理..."
    echo "=========================================="
    
    PLANS_FILE="${MODEL_DIR}/plans.json"
    DATASET_JSON="${MODEL_DIR}/dataset.json"
    
    if [ -f "${PLANS_FILE}" ] && [ -f "${DATASET_JSON}" ]; then
        POSTPROCESSED_OUTPUT="${OUTPUT_DIR}_postprocessed"
        mkdir -p "${POSTPROCESSED_OUTPUT}"
        
        nnUNetv2_apply_postprocessing \
            -i "${OUTPUT_DIR}" \
            -o "${POSTPROCESSED_OUTPUT}" \
            --pp_pkl_file "${POSTPROCESSING_FILE}" \
            -plans_json "${PLANS_FILE}" \
            -dataset_json "${DATASET_JSON}"
        
        if [ $? -eq 0 ]; then
            echo "✅ 后处理完成！结果保存在: ${POSTPROCESSED_OUTPUT}"
        fi
    fi
fi

echo ""
echo "完成！"



