# Convert 2D Retinal Fundus Images to nnUNet Format

This script converts 2D retinal fundus images and segmentation masks into nnUNet format.

## Data Requirements

- **Image folder**: Training images (.png, .jpg, .tif, etc.)
- **Mask folder**: Segmentation masks (matched by filename)

### Filename Matching

1. **Exact**: `image_001.png` ↔ `image_001.png`
2. **Suffix**: `image_001.png` ↔ `image_001_mask.png` or `image_001_gt.png`
3. **Contains**: Image name contained in mask name or vice versa

## Usage

### Option 1: Command line

```bash
python convert_retinal_fundus_to_nnunet.py \
    --images_dir /path/to/your/images \
    --masks_dir /path/to/your/masks \
    --dataset_id 001 \
    --dataset_name RetinalFundus
```

### 方法2：修改示例脚本

1. 编辑 `convert_example.sh`，修改路径：
```bash
IMAGES_DIR="/path/to/your/images"
MASKS_DIR="/path/to/your/masks"
DATASET_ID="001"
DATASET_NAME="RetinalFundus"
```

2. 运行脚本：
```bash
bash convert_example.sh
```

## 参数说明

- `--images_dir`: 图像文件夹路径（必需）
- `--masks_dir`: mask文件夹路径（必需）
- `--dataset_id`: 数据集ID，3位数字（必需，如 001, 002）
- `--dataset_name`: 数据集名称（必需，如 RetinalFundus）
- `--output_dir`: nnUNet_raw输出路径（可选，默认使用环境变量）
- `--file_ending`: 输出文件扩展名（可选，默认.png）

## 输出结构

转换后，数据将按照nnUNet格式组织：

```
nnUNet_raw/
└── Dataset001_RetinalFundus/
    ├── dataset.json          # 数据集元数据（自动生成）
    ├── imagesTr/              # 训练图像
    │   ├── case001_0000.png
    │   ├── case002_0000.png
    │   └── ...
    └── labelsTr/              # 训练标签
        ├── case001.png
        ├── case002.png
        └── ...
```

## 自动功能

脚本会自动：
1. ✅ **匹配图像和mask文件**：智能匹配文件名
2. ✅ **分析mask标签**：自动检测mask中的所有类别
3. ✅ **标签重映射**：确保标签连续（0, 1, 2, ...）
4. ✅ **格式转换**：将图像转换为PNG格式（无损）
5. ✅ **生成dataset.json**：自动创建nnUNet所需的配置文件

## 注意事项

1. **标签要求**：
   - 背景必须是0
   - 其他类别必须是连续的整数（1, 2, 3, ...）
   - 如果mask中有非连续标签（如0, 1, 255），脚本会自动重映射

2. **文件格式**：
   - 支持输入：.png, .jpg, .jpeg, .tif, .tiff
   - 输出格式：.png（无损压缩）

3. **Mask格式**：
   - 如果是RGB mask，脚本会自动提取第一个通道
   - 确保mask是整数类型

## 转换后的下一步

转换完成后，按照以下步骤继续：

### 1. 设置环境变量（如果还没设置）

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### 2. 运行实验规划和预处理

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

### 3. 开始训练

```bash
# 训练2D U-Net（5折交叉验证）
nnUNetv2_train 001 2d 0 --npz  # fold 0
nnUNetv2_train 001 2d 1 --npz  # fold 1
nnUNetv2_train 001 2d 2 --npz  # fold 2
nnUNetv2_train 001 2d 3 --npz  # fold 3
nnUNetv2_train 001 2d 4 --npz  # fold 4
```

### 4. 找到最佳配置

```bash
nnUNetv2_find_best_configuration 001 -c 2d
```

### 5. 运行推理

```bash
nnUNetv2_predict -i /path/to/test/images -o /path/to/output -d 001 -c 2d
```

## 常见问题

### Q: 图像和mask文件名不匹配怎么办？

A: 脚本会尝试多种匹配方式。如果仍然无法匹配，请检查：
- 文件名是否包含相同的标识符
- 文件扩展名是否正确
- 确保图像和mask数量一致

### Q: mask中有多个类别，但标签值不连续怎么办？

A: 脚本会自动检测并重映射标签，确保标签连续（0, 1, 2, ...）

### Q: 如何查看转换后的dataset.json？

A: 转换完成后，可以在 `nnUNet_raw/DatasetXXX_Name/dataset.json` 中查看配置信息。

### Q: 可以使用其他数据集ID吗？

A: 可以，但确保不与已有数据集冲突。建议从001开始，或查看 `nnUNet_raw` 中已有的数据集ID。

## 示例

假设你有以下数据：
```
my_data/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── img_003.jpg
└── masks/
    ├── img_001.png
    ├── img_002.png
    └── img_003.png
```

运行转换：
```bash
python convert_retinal_fundus_to_nnunet.py \
    --images_dir my_data/images \
    --masks_dir my_data/masks \
    --dataset_id 001 \
    --dataset_name MyRetinalDataset
```

转换后，数据将组织为：
```
nnUNet_raw/Dataset001_MyRetinalDataset/
├── dataset.json
├── imagesTr/
│   ├── img_001_0000.png
│   ├── img_002_0000.png
│   └── img_003_0000.png
└── labelsTr/
    ├── img_001.png
    ├── img_002.png
    └── img_003.png
```



