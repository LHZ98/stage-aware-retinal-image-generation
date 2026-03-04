#!/bin/bash

# Train both: FIVE (original) and FIVE_CROP (preprocessed)
# FIVE on CUDA:0, FIVE_CROP on CUDA:1; 5-fold CV, fold 0 only

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Start both training jobs"
echo "=========================================="
echo ""
echo "Job 1: FIVE (original) - CUDA:0 - Fold 0"
echo "Job 2: FIVE_CROP (Circle Cropping) - CUDA:1 - Fold 0"
echo "Config: 5-fold CV, fold 0 only, 1000 epochs"
echo ""

LOG_DIR="${SCRIPT_DIR}/training_logs"
mkdir -p "${LOG_DIR}"

echo "Starting FIVE (original)..."
nohup bash train_FIVE_original.sh > "${LOG_DIR}/FIVE_original_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID1=$!
echo "FIVE started PID: ${PID1}, log: ${LOG_DIR}/FIVE_original_*.log"
echo ""

sleep 5

echo "Starting FIVE_CROP (Circle Cropping)..."
nohup bash train_FIVE_crop.sh > "${LOG_DIR}/FIVE_crop_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID2=$!
echo "FIVE_CROP started PID: ${PID2}, log: ${LOG_DIR}/FIVE_crop_*.log"
echo ""

echo "=========================================="
echo "Both jobs started"
echo "=========================================="
echo ""
echo "PIDs: FIVE ${PID1}, FIVE_CROP ${PID2}"
echo ""
echo "Config: 5-fold CV, Fold 0, 1000 epochs, checkpoint every 50"
echo ""
echo "Progress: tail -f ${LOG_DIR}/FIVE_original_*.log"
echo "          tail -f ${LOG_DIR}/FIVE_crop_*.log"
echo ""
echo "Check: ps aux | grep ${PID1}"
echo "  ps aux | grep ${PID2}"
echo ""
echo "Stop: kill ${PID1}  # FIVE"
echo "      kill ${PID2}  # FIVE_CROP"
echo ""
