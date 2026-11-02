# ParaDis-SCD: Disocclusion-Aware Epipolar Geometry and Dynamic Homography for Robust Street-View Semantic Change Detection

Official PyTorch implementation of **ParaDis-SCD** for street-view and ground robotics semantic change detection (SCD).

---

## Overview

Street-view semantic change detection is prone to large viewpoint disparities, acute baseline shifts, and severe 3D parallax occlusions. Traditional 2D differencing and attention operators frequently mistake newly exposed backgrounds (**disocclusions**) for genuine physical changes. 

**ParaDis-SCD** explicitly disentangles parallax-driven disocclusions from genuine physical scene changes by unifying multi-view epipolar geometry, dynamic ground homography, and physics-grounded orthogonal gating.

### Key Innovations
1. **Dynamic Bilateral Appearance & Illumination Alignment (`D-BAIA`)**: Physics-grounded Retinex decomposition and bilateral alignment to eliminate volatile illumination drifts and weather disparities.
2. **Disocclusion-Aware Epipolar Gauge (`D-ESG`)**: Imposes first-order epipolar ray geometric constraints via fundamental matrix geometry, bounding feature search along epipolar manifolds.
3. **Epipolar-Constrained Deformable Cross-Attention (`EC-DCA`)**: Samples feature offsets strictly along epipolar rays with bounded uncertainty windows.
4. **Projective Disocclusion-Aware Gating (`P-DAG`)**: Dynamically decouples occlusion boundaries from physical structural variations via null-space orthogonal projection.

---

## Repository Structure

```text
ParaDis-SCD/
├── paradis_scd/
│   ├── models/
│   │   ├── paradis_scd.py          # Main ParaDis-SCD network architecture
│   │   ├── backbone.py             # Multi-scale hierarchical feature extractor
│   │   ├── retinex_align.py        # Dynamic Bilateral Appearance Alignment (D-BAIA)
│   │   ├── d_esg.py                # Disocclusion-Aware Epipolar Gauge (D-ESG)
│   │   ├── ec_deform_attn.py       # Epipolar-Constrained Deformable Cross-Attention (EC-DCA)
│   │   ├── p_dag.py                # Projective Disocclusion-Aware Gating (P-DAG)
│   │   ├── mg_priors.py            # Multiview geometric invariants (SO(3), SL(3))
│   │   └── decoder.py              # Multi-scale boundary-preserving change decoder
│   ├── datasets/                   # Real-world benchmark loaders (PCD, ChangeSim, VL-CMU-CD)
│   ├── losses/                     # 6 physics & boundary loss formulations
│   └── metrics.py                  # Analytical SCD evaluation metrics (F1, IoU, Precision, Recall)
├── baselines/                      # Unified comparison baselines
│   ├── c3po.py                     # C-3PO (CVPR 2023 / PR)
│   ├── changer.py                  # Changer (ICCV 2023)
│   ├── csnet.py                    # CSNet (ICRA 2021)
│   ├── dr_tanet.py                 # DR-TANet (IEEE TIP 2020)
│   ├── ussfcnet.py                 # USSFC-Net (IEEE TGRS 2023)
│   ├── changeformer.py             # ChangeFormer (IEEE J-STARS 2022)
│   └── cgnet.py                    # CGNet (IEEE J-STARS 2023)
├── train.py                        # Unified training entry
├── eval.py                         # Single-model evaluation script
├── verify_paradis_scd.py           # Master geometric & analytical verification suite
├── summarize_benchmark.py          # Multi-model benchmark comparison table generator
├── run_real_scd_benchmark.sh       # Full benchmark execution pipeline
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md
```

---

## Getting Started

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n paradis python=3.10 -y
conda activate paradis

# Install dependencies
pip install -r requirements.txt
```

### 2. Geometric Verification Suite

Run our self-contained master verification suite to test SO(3) rotations, epipolar constraints, and arbitrary resolution adapters:

```bash
python verify_paradis_scd.py
```

### 3. Training & Evaluation

#### Training ParaDis-SCD:
```bash
python -u train.py \
    --model paradis_scd \
    --dataset pcd \
    --data_dir /path/to/PCD \
    --img_size 224 1024 \
    --epochs 15 \
    --batch_size 2 \
    --lr 1e-4 \
    --amp \
    --save_dir ./checkpoints/pcd_paradis_scd
```

#### Evaluating Checkpoint:
```bash
python -u eval.py \
    --model paradis_scd \
    --dataset pcd \
    --data_dir /path/to/PCD \
    --img_size 224 1024 \
    --resume ./checkpoints/pcd_paradis_scd/best.pth \
    --eval_dir ./eval_results/pcd_paradis_scd
```

#### Running Benchmark Baselines:
```bash
bash run_real_scd_benchmark.sh
```

---

## Supported Datasets

- **PCD (TSUNAMI / GSV)**: Panoramic street-view change detection benchmark with extreme lateral baseline shifts.
- **ChangeSim**: Ground industrial multi-hazard environment dataset featuring drastic viewpoint disparities and dynamic air turbidity.
- **VL-CMU-CD**: Long-term vehicle-mounted autonomous driving dataset across diverse weather and seasonal cycles.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
