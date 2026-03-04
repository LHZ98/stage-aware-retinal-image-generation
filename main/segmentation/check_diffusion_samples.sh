#!/bin/bash
# Check intermediate samples from Segmentation-Guided Diffusion training

OUTPUT_DIR="/media/yuganlab/blackstone/MICCAI2026/segmentation-guided-diffusion/ddim-aptos2019-256-segguided"
SAMPLES_DIR="${OUTPUT_DIR}/samples"

echo "=========================================="
echo "Check Diffusion training samples"
echo "=========================================="
echo "Output dir: ${OUTPUT_DIR}"
echo "Samples dir: ${SAMPLES_DIR}"
echo ""

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Output dir not created yet (training may have just started)"
    echo "Dir will be created after first epoch"
    exit 0
fi

echo "Output dir exists"
echo ""

if [ ! -d "${SAMPLES_DIR}" ]; then
    echo "Samples dir not created yet"
    echo "Samples saved every 20 epochs (save_image_epochs=20)"
    echo "First sample at Epoch 20"
    exit 0
fi

echo "Samples dir exists"
echo ""
echo "Sample files:"
ls -lh "${SAMPLES_DIR}" | tail -20

echo ""
echo "Sample stats:"
echo "  Generated: $(ls -1 ${SAMPLES_DIR}/*.png 2>/dev/null | grep -v cond | grep -v orig | wc -l)"
echo "  Cond masks: $(ls -1 ${SAMPLES_DIR}/*cond*.png 2>/dev/null | wc -l)"
echo "  Original: $(ls -1 ${SAMPLES_DIR}/*orig*.png 2>/dev/null | wc -l)"
echo ""

LATEST_SAMPLE=$(ls -t ${SAMPLES_DIR}/*.png 2>/dev/null | grep -v cond | grep -v orig | head -1)
if [ -n "${LATEST_SAMPLE}" ]; then
    echo "Latest sample:"
    echo "  $(basename ${LATEST_SAMPLE})"
    echo "  Size: $(ls -lh ${LATEST_SAMPLE} | awk '{print $5}')"
    echo "  Modified: $(ls -l ${LATEST_SAMPLE} | awk '{print $6, $7, $8}')"
fi

echo ""
echo "=========================================="
echo "Notes:"
echo "  - Samples saved every 20 epochs"
echo "  - Format: {epoch:04d}.png (generated)"
echo "  - Format: {epoch:04d}_cond_{seg_type}.png (cond mask)"
echo "  - Format: {epoch:04d}_orig.png (original)"
echo "=========================================="
