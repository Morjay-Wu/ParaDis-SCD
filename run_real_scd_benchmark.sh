#!/usr/bin/env bash
# Sequential Real-Dataset Training & Evaluation Benchmark for ParaDis-SCD & 2023-2024 SOTA Baselines
# Designed for NVIDIA A40 shared environment (< 3.8 GB VRAM per process)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================================================"
echo "Starting Real-Dataset SCD Benchmark at $(date)"
echo "Working Directory: $SCRIPT_DIR"
echo "======================================================================"

# Step 0: Verify real datasets are loaded (is_synthetic == False)
python -u verify_real_datasets.py

# Step 1: Sequential Training & Evaluation on PCD (224x1024, real TSUNAMI dataset)
PCD_MODELS=("paradis_scd" "c3po" "cgnet" "ussfcnet" "changer" "csnet" "changeformer" "dr_tanet")
for MODEL in "${PCD_MODELS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "[PCD 224x1024] Training model: ${MODEL}"
    echo "----------------------------------------------------------------------"
    SAVE_DIR="./checkpoints_real/pcd_${MODEL}"
    EVAL_DIR="./eval_results_real/pcd_${MODEL}"
    python -u train.py \
        --model "${MODEL}" \
        --dataset pcd \
        --data_dir /home/WUtongquan/dataset/PCD \
        --img_size 224 1024 \
        --epochs 15 \
        --batch_size 1 \
        --lr 1e-4 \
        --amp \
        --save_dir "${SAVE_DIR}"

    python -u eval.py \
        --model "${MODEL}" \
        --dataset pcd \
        --data_dir /home/WUtongquan/dataset/PCD \
        --split test \
        --img_size 224 1024 \
        --batch_size 1 \
        --weights "${SAVE_DIR}/best.pth" \
        --output_dir "${EVAL_DIR}" \
        --save_vis --vis_num 5
done

# Step 2: Sequential Training & Evaluation on ChangeSim (256x384 / 480x640, real warehouse dataset)
CS_MODELS=("paradis_scd" "c3po" "cgnet" "ussfcnet" "changer" "csnet" "changeformer" "dr_tanet")
for MODEL in "${CS_MODELS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "[ChangeSim 256x384] Training model: ${MODEL}"
    echo "----------------------------------------------------------------------"
    SAVE_DIR="./checkpoints_real/changesim_${MODEL}"
    EVAL_DIR="./eval_results_real/changesim_${MODEL}"
    python -u train.py \
        --model "${MODEL}" \
        --dataset changesim \
        --data_dir /home/WUtongquan/dataset/ChangeSim \
        --img_size 256 384 \
        --epochs 15 \
        --batch_size 2 \
        --lr 2e-4 \
        --amp \
        --save_dir "${SAVE_DIR}"

    python -u eval.py \
        --model "${MODEL}" \
        --dataset changesim \
        --data_dir /home/WUtongquan/dataset/ChangeSim \
        --split test \
        --img_size 256 384 \
        --batch_size 2 \
        --weights "${SAVE_DIR}/best.pth" \
        --output_dir "${EVAL_DIR}" \
        --save_vis --vis_num 5
done

# Step 3: Sequential Training & Evaluation on VL-CMU-CD (256x384, real street-view dataset)
VL_MODELS=("paradis_scd" "c3po" "cgnet" "ussfcnet" "changer" "csnet" "changeformer" "dr_tanet")
for MODEL in "${VL_MODELS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "[VL-CMU-CD 256x384] Training model: ${MODEL}"
    echo "----------------------------------------------------------------------"
    SAVE_DIR="./checkpoints_real/vlcmu_${MODEL}"
    EVAL_DIR="./eval_results_real/vlcmu_${MODEL}"
    python -u train.py \
        --model "${MODEL}" \
        --dataset vlcmu \
        --data_dir /home/WUtongquan/dataset/VL-CMU-CD \
        --img_size 256 384 \
        --epochs 10 \
        --batch_size 2 \
        --lr 2e-4 \
        --amp \
        --save_dir "${SAVE_DIR}"

    python -u eval.py \
        --model "${MODEL}" \
        --dataset vlcmu \
        --data_dir /home/WUtongquan/dataset/VL-CMU-CD \
        --split test \
        --img_size 256 384 \
        --batch_size 2 \
        --weights "${SAVE_DIR}/best.pth" \
        --output_dir "${EVAL_DIR}" \
        --save_vis --vis_num 5
done

echo "======================================================================"
echo "All Real-Dataset SCD Benchmark runs completed at $(date)!"
echo "======================================================================"
