#!/bin/bash
# Create and set up seg_diff conda environment

set -e

echo "=========================================="
echo "Create seg_diff conda environment"
echo "=========================================="

if conda env list | grep -q "seg_diff"; then
    echo "Environment seg_diff exists, skip create"
else
    echo "Creating conda env: seg_diff (Python 3.11)"
    conda create -n seg_diff python=3.11 -y
fi

echo ""
echo "Activate and install dependencies..."
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate seg_diff

echo "Python: $(python --version)"
echo ""

echo "Check CUDA version..."
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
echo "CUDA: ${CUDA_VERSION}"

echo ""
echo "Install PyTorch..."
if [[ "${CUDA_VERSION}" == "12."* ]]; then
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
elif [[ "${CUDA_VERSION}" == "11."* ]]; then
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
else
    echo "Using default PyTorch..."
    pip install torch==2.1.2 torchvision==0.16.2
fi

echo ""
echo "Install requirements..."
cd /media/yuganlab/blackstone/MICCAI2026/segmentation-guided-diffusion
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Environment ready."
echo "=========================================="
echo ""
echo "Activate: conda activate seg_diff"
echo ""
echo "Verify: python -c 'import torch; import diffusers; print(torch.__version__, diffusers.__version__)'"
echo ""
