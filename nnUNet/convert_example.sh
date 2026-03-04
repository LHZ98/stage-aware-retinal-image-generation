#!/bin/bash
# 使用示例脚本 - 将你的图像和mask转换为nnUNet格式

# ============================================
# 请修改以下路径为你的实际路径
# ============================================

# 你的图像文件夹路径
IMAGES_DIR="/path/to/your/images"

# 你的mask文件夹路径
MASKS_DIR="/path/to/your/masks"

# 数据集ID（3位数字，如001, 002等，不要与已有数据集冲突）
DATASET_ID="001"

# 数据集名称（可以自定义，如 RetinalFundus, CHASE_DB1等）
DATASET_NAME="RetinalFundus"

# nnUNet_raw输出路径（如果设置了环境变量nnUNet_raw，可以留空）
OUTPUT_DIR=""  # 留空则使用环境变量nnUNet_raw

# ============================================
# 执行转换
# ============================================

python convert_retinal_fundus_to_nnunet.py \
    --images_dir "$IMAGES_DIR" \
    --masks_dir "$MASKS_DIR" \
    --dataset_id "$DATASET_ID" \
    --dataset_name "$DATASET_NAME" \
    ${OUTPUT_DIR:+--output_dir "$OUTPUT_DIR"}



