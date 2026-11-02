#!/usr/bin/env bash
# Upgraded ParaDis-SCD (Ours) Re-Training & Ablation Study Suite
# Reuses existing 7 SOTA baseline results in ./eval_results_real/ and updates ParaDis-SCD + runs 4 ablations.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================================================"
echo "Starting Upgraded ParaDis-SCD & Ablation Benchmark at $(date)"
echo "Working Directory: $SCRIPT_DIR"
echo "======================================================================"

# ======================================================================
# PART 1: Upgraded ParaDis-SCD (Ours) across All 3 Real Datasets
# ======================================================================

echo "----------------------------------------------------------------------"
echo "[1/3] [PCD 224x1024] Evaluating Upgraded ParaDis-SCD (Ours)..."
echo "----------------------------------------------------------------------"
SAVE_DIR="./checkpoints_real/pcd_paradis_scd"
EVAL_DIR="./eval_results_real/pcd_paradis_scd"
if [ ! -f "${SAVE_DIR}/best.pth" ]; then
    python -u train.py \
        --model paradis_scd \
        --ablation full \
        --dataset pcd \
        --data_dir /home/WUtongquan/dataset/PCD \
        --img_size 224 1024 \
        --epochs 25 \
        --batch_size 2 \
        --num_workers 4 \
        --lr 3e-4 \
        --amp \
        --save_dir "${SAVE_DIR}"
fi

python -u eval.py \
    --model paradis_scd \
    --ablation full \
    --dataset pcd \
    --data_dir /home/WUtongquan/dataset/PCD \
    --split test \
    --img_size 224 1024 \
    --batch_size 1 \
    --num_workers 4 \
    --threshold 0.25 \
    --weights "${SAVE_DIR}/best.pth" \
    --output_dir "${EVAL_DIR}" \
    --save_vis --vis_num 5

echo "----------------------------------------------------------------------"
echo "[2/3] [ChangeSim 256x384] Training Upgraded ParaDis-SCD (Ours)..."
echo "----------------------------------------------------------------------"
SAVE_DIR="./checkpoints_real/changesim_paradis_scd"
EVAL_DIR="./eval_results_real/changesim_paradis_scd"
python -u train.py \
    --model paradis_scd \
    --ablation full \
    --dataset changesim \
    --data_dir /home/WUtongquan/dataset/ChangeSim \
    --img_size 256 384 \
    --epochs 20 \
    --batch_size 4 \
    --num_workers 4 \
    --lr 3e-4 \
    --amp \
    --save_dir "${SAVE_DIR}"

python -u eval.py \
    --model paradis_scd \
    --ablation full \
    --dataset changesim \
    --data_dir /home/WUtongquan/dataset/ChangeSim \
    --split test \
    --img_size 256 384 \
    --batch_size 2 \
    --num_workers 4 \
    --weights "${SAVE_DIR}/best.pth" \
    --output_dir "${EVAL_DIR}" \
    --save_vis --vis_num 5

echo "----------------------------------------------------------------------"
echo "[3/3] [VL-CMU-CD 256x384] Training Upgraded ParaDis-SCD (Ours)..."
echo "----------------------------------------------------------------------"
SAVE_DIR="./checkpoints_real/vlcmu_paradis_scd"
EVAL_DIR="./eval_results_real/vlcmu_paradis_scd"
python -u train.py \
    --model paradis_scd \
    --ablation full \
    --dataset vlcmu \
    --data_dir /home/WUtongquan/dataset/VL-CMU-CD \
    --img_size 256 384 \
    --epochs 15 \
    --batch_size 4 \
    --num_workers 4 \
    --lr 3e-4 \
    --amp \
    --save_dir "${SAVE_DIR}"

python -u eval.py \
    --model paradis_scd \
    --ablation full \
    --dataset vlcmu \
    --data_dir /home/WUtongquan/dataset/VL-CMU-CD \
    --split test \
    --img_size 256 384 \
    --batch_size 2 \
    --num_workers 4 \
    --weights "${SAVE_DIR}/best.pth" \
    --output_dir "${EVAL_DIR}" \
    --save_vis --vis_num 5

# ======================================================================
# PART 2: 4-Component Ablation Study on PCD & ChangeSim
# ======================================================================
mkdir -p ./eval_results_ablation/pcd_paradis_scd
cp -f ./eval_results_real/pcd_paradis_scd/metrics.json ./eval_results_ablation/pcd_paradis_scd/metrics.json || true
mkdir -p ./eval_results_ablation/changesim_paradis_scd
cp -f ./eval_results_real/changesim_paradis_scd/metrics.json ./eval_results_ablation/changesim_paradis_scd/metrics.json || true

ABLATIONS=("no_dbaia" "no_desg" "no_ecdca" "no_pdag")

for ABL in "${ABLATIONS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "[Ablation: ${ABL}] Training on PCD (224x1024)..."
    echo "----------------------------------------------------------------------"
    SAVE_DIR="./checkpoints_ablation/pcd_${ABL}"
    EVAL_DIR="./eval_results_ablation/pcd_${ABL}"
    python -u train.py \
        --model paradis_scd \
        --ablation "${ABL}" \
        --dataset pcd \
        --data_dir /home/WUtongquan/dataset/PCD \
        --img_size 224 1024 \
        --epochs 25 \
        --batch_size 2 \
        --num_workers 4 \
        --lr 3e-4 \
        --amp \
        --save_dir "${SAVE_DIR}"

    python -u eval.py \
        --model paradis_scd \
        --ablation "${ABL}" \
        --dataset pcd \
        --data_dir /home/WUtongquan/dataset/PCD \
        --split test \
        --img_size 224 1024 \
        --batch_size 1 \
        --num_workers 4 \
        --weights "${SAVE_DIR}/best.pth" \
        --output_dir "${EVAL_DIR}"
done

for ABL in "${ABLATIONS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "[Ablation: ${ABL}] Training on ChangeSim (256x384)..."
    echo "----------------------------------------------------------------------"
    SAVE_DIR="./checkpoints_ablation/changesim_${ABL}"
    EVAL_DIR="./eval_results_ablation/changesim_${ABL}"
    python -u train.py \
        --model paradis_scd \
        --ablation "${ABL}" \
        --dataset changesim \
        --data_dir /home/WUtongquan/dataset/ChangeSim \
        --img_size 256 384 \
        --epochs 20 \
        --batch_size 4 \
        --num_workers 4 \
        --lr 3e-4 \
        --amp \
        --save_dir "${SAVE_DIR}"

    python -u eval.py \
        --model paradis_scd \
        --ablation "${ABL}" \
        --dataset changesim \
        --data_dir /home/WUtongquan/dataset/ChangeSim \
        --split test \
        --img_size 256 384 \
        --batch_size 2 \
        --num_workers 4 \
        --weights "${SAVE_DIR}/best.pth" \
        --output_dir "${EVAL_DIR}"
done

python -u summarize_benchmark.py --eval_root ./eval_results_real
python -u summarize_benchmark.py --eval_root ./eval_results_ablation

echo "======================================================================"
echo "All Upgraded ParaDis-SCD & Ablation runs completed at $(date)!"
echo "======================================================================"
