"""Unified Evaluation Script for Scene Change Detection (SCD).

Capabilities:
1. Evaluates any supported model ('paradis_scd', 'csnet', 'changeformer', 'dr_tanet')
   on any supported benchmark ('changesim', 'vlcmu', 'pcd').
2. Computes 7-metric evaluation via SCDMetrics:
   - Overall Accuracy (OA)
   - Change F1-Score
   - Mean IoU (mIoU)
   - Change IoU
   - Static IoU
   - Precision
   - Recall
3. Generates qualitative visual change detection maps:
   5-panel side-by-side composite:
   [Image 1 (T1)] | [Image 2 (T2)] | [Ground Truth Mask] | [Predicted Change Mask] | [Color-Coded Error Map]
   Error Map color scheme:
   - Green [0, 255, 0]      : True Positive (TP - Correct Change Detection)
   - Red   [255, 0, 0]      : False Positive (FP - False Alarm)
   - Blue  [0, 120, 255]    : False Negative (FN - Missed Change)
   - Dark  [30, 30, 30]     : True Negative (TN - Correct Static Background)
4. Saves metrics.json, metrics.txt, and visualization PNGs in output directory.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

# Ensure cvpr2 is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paradis_scd.models import ParaDisSCDNet
from paradis_scd.metrics import SCDMetrics
from paradis_scd.datasets import (
    ChangeSimDataset,
    VLCMUDataset,
    PCDDataset,
)
from baselines import get_baseline_model


DEFAULT_DATASET_DIRS = {
    "changesim": "/home/WUtongquan/dataset/ChangeSim",
    "vlcmu": "/home/WUtongquan/dataset/VL-CMU-CD",
    "pcd": "/home/WUtongquan/dataset/PCD",
}

SUPPORTED_MODELS = [
    "paradis_scd",
    "csnet",
    "changeformer",
    "dr_tanet",
    "c3po",
    "cgnet",
    "ussfcnet",
    "changer",
]


def get_dataset(
    name: str,
    split: str,
    root_dir: Optional[str] = None,
    img_size: Tuple[int, int] = (480, 640),
):
    name_clean = name.lower().replace("-", "_").replace(" ", "")
    if name_clean in ("changesim", "changesim_dataset"):
        if root_dir is None and os.path.isdir(DEFAULT_DATASET_DIRS["changesim"]):
            root_dir = DEFAULT_DATASET_DIRS["changesim"]
        return ChangeSimDataset(root_dir=root_dir, split=split, img_size=img_size)
    elif name_clean in ("vlcmu", "vl_cmu", "vlcmu_cd", "vl_cmu_cd"):
        if root_dir is None and os.path.isdir(DEFAULT_DATASET_DIRS["vlcmu"]):
            root_dir = DEFAULT_DATASET_DIRS["vlcmu"]
        return VLCMUDataset(root_dir=root_dir, split=split, img_size=img_size)
    elif name_clean in ("pcd", "pcd_dataset"):
        if root_dir is None and os.path.isdir(DEFAULT_DATASET_DIRS["pcd"]):
            root_dir = DEFAULT_DATASET_DIRS["pcd"]
        return PCDDataset(root_dir=root_dir, split=split, img_size=img_size)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: ['changesim', 'vlcmu', 'pcd']")


def get_model(name: str, ablation: str = "full"):
    name_clean = name.lower().replace("-", "_").replace(" ", "")
    if name_clean in ("paradis_scd", "paradis", "paradisscd"):
        return ParaDisSCDNet(pretrained=False, freeze_priors=False, ablation=ablation)
    else:
        return get_baseline_model(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SCD Evaluation & Visualization Script")
    parser.add_argument(
        "--model",
        type=str,
        default="paradis_scd",
        choices=SUPPORTED_MODELS,
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pth). If None, evaluates freshly initialized weights.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="changesim",
        choices=["changesim", "vlcmu", "pcd"],
        help="Dataset benchmark",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for dataset")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory for results")
    parser.add_argument("--save_vis", action="store_true", help="Generate and save qualitative visualization maps")
    parser.add_argument("--vis_num", type=int, default=10, help="Maximum number of visualizations to save")
    parser.add_argument("--batch_size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--img_size", type=int, nargs=2, default=[480, 640], help="Image height and width")
    parser.add_argument("--threshold", type=float, default=0.5, help="Change probability decision threshold")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_dbaia", "no_desg", "no_ecdca", "no_pdag"],
        help="Ablation mode for ParaDis-SCD",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device",
    )
    return parser.parse_args()


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized or unnormalized image tensor [3, H, W] to uint8 RGB array [H, W, 3]."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    # Check if normalized with ImageNet stats or in [0, 1]
    if arr.min() < 0.0 or arr.max() <= 1.05:
        # Standard un-normalization assuming [0, 1] or ImageNet
        if arr.min() < 0.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            arr = (arr * std + mean) * 255.0
        else:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def create_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Creates color-coded error map: TP=Green, FP=Red, FN=Blue, TN=Dark Gray.
    
    Args:
        pred: Boolean array [H, W] of predicted change
        gt: Boolean array [H, W] of ground truth change
    """
    H, W = pred.shape
    error_map = np.zeros((H, W, 3), dtype=np.uint8)

    # True Negative (Static correctly classified)
    tn = (~pred) & (~gt)
    error_map[tn] = [30, 30, 30]

    # True Positive (Change correctly detected) -> Green
    tp = pred & gt
    error_map[tp] = [0, 255, 0]

    # False Positive (False Alarm) -> Red
    fp = pred & (~gt)
    error_map[fp] = [255, 0, 0]

    # False Negative (Missed Change) -> Blue
    fn = (~pred) & gt
    error_map[fn] = [0, 120, 255]

    return error_map


def build_composite_visualization(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    name: str = "sample",
) -> Image.Image:
    """Builds 5-panel side-by-side composite: [Img1 | Img2 | GT | Pred | Error Map]."""
    H, W, _ = img1_rgb.shape

    # Binary masks as 3-channel RGB
    gt_vis = np.stack([gt_mask.astype(np.uint8) * 255] * 3, axis=-1)
    pred_vis = np.stack([pred_mask.astype(np.uint8) * 255] * 3, axis=-1)

    # Color-coded error map
    error_map = create_error_map(pred_mask, gt_mask)

    # Concatenate horizontally
    panel_gap = 4
    total_w = W * 5 + panel_gap * 4
    header_h = 36
    total_h = H + header_h

    composite = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(composite)

    # Paste panels
    composite.paste(Image.fromarray(img1_rgb), (0, header_h))
    composite.paste(Image.fromarray(img2_rgb), (W + panel_gap, header_h))
    composite.paste(Image.fromarray(gt_vis), ((W + panel_gap) * 2, header_h))
    composite.paste(Image.fromarray(pred_vis), ((W + panel_gap) * 3, header_h))
    composite.paste(Image.fromarray(error_map), ((W + panel_gap) * 4, header_h))

    # Titles
    titles = [
        "Image 1 (Time 1)",
        "Image 2 (Time 2)",
        "Ground Truth Mask",
        "Predicted Change Mask",
        "Error Map (TP:Grn, FP:Red, FN:Blu)",
    ]
    for i, title in enumerate(titles):
        x = i * (W + panel_gap) + 10
        draw.text((x, 10), title, fill=(240, 240, 240))

    return composite


def main():
    args = parse_args()
    device = torch.device(args.device)
    target_img_size = (args.img_size[0], args.img_size[1])

    print("=" * 70)
    print("SCD COMPREHENSIVE EVALUATION & BENCHMARKING")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : {args.dataset} (split: {args.split})")
    print(f"  Resolution  : {target_img_size[0]}x{target_img_size[1]}")
    print(f"  Weights     : {args.weights or '[Uninitialized/Fresh]'}")
    print(f"  Device      : {device}")
    print(f"  Save Vis    : {args.save_vis} (max {args.vis_num})")
    print(f"  Output Dir  : {args.output_dir}")
    print("-" * 70)

    # 1. Dataset & DataLoader
    eval_dataset = get_dataset(
        name=args.dataset,
        split=args.split,
        root_dir=args.data_dir,
        img_size=target_img_size,
    )
    print(f"  Eval Data   : {len(eval_dataset)} samples (is_synthetic={eval_dataset.is_synthetic}, root={eval_dataset.root_dir})")
    print("-" * 70)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model Loading
    model = get_model(args.model, ablation=args.ablation).to(device)
    if args.weights:
        if os.path.isfile(args.weights):
            print(f"Loading checkpoint weights from: {args.weights}")
            ckpt = torch.load(args.weights, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print("Checkpoint weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args.weights}")

    model.eval()

    # 3. Output directory setup
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    metrics_tracker = SCDMetrics(threshold=args.threshold)
    saved_vis_count = 0

    print("Running evaluation across dataset...")
    t_start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            names = batch.get("name", [f"sample_{step}_{i}" for i in range(len(img1))])

            outputs = model(img1, img2)
            logits = outputs["logits"]
            probs = outputs.get("prob", torch.sigmoid(logits))

            # Cumulative metric update
            metrics_tracker.update(logits, mask)

            # Qualitative visual change map creation
            if args.save_vis and saved_vis_count < args.vis_num:
                B = img1.shape[0]
                for b in range(B):
                    if saved_vis_count >= args.vis_num:
                        break
                    img1_rgb = tensor_to_rgb_image(img1[b])
                    img2_rgb = tensor_to_rgb_image(img2[b])
                    gt_b = (mask[b, 0].detach().cpu().numpy() > 0.5)
                    pred_b = (probs[b, 0].detach().cpu().numpy() >= args.threshold)
                    sample_name = names[b] if isinstance(names, (list, tuple)) else f"sample_{saved_vis_count}"

                    composite_img = build_composite_visualization(
                        img1_rgb=img1_rgb,
                        img2_rgb=img2_rgb,
                        gt_mask=gt_b,
                        pred_mask=pred_b,
                        name=sample_name,
                    )
                    clean_name = os.path.basename(str(sample_name)).replace(".png", "").replace(".jpg", "")
                    vis_path = os.path.join(vis_dir, f"{saved_vis_count:03d}_{clean_name}_vis.png")
                    composite_img.save(vis_path)
                    saved_vis_count += 1

    eval_duration = time.time() - t_start
    results = metrics_tracker.compute()

    # Formatted comparison summary table
    summary_str = metrics_tracker.summary()
    print("\n" + summary_str)
    print(f"Evaluation completed in {eval_duration:.2f}s.")
    if args.save_vis:
        print(f"Generated {saved_vis_count} qualitative change maps in: {vis_dir}")

    # Save artifacts to output directory
    model_tag = args.model if args.ablation == "full" else f"{args.model}_{args.ablation}"
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": model_tag,
                "ablation": args.ablation,
                "dataset": args.dataset,
                "split": args.split,
                "resolution": target_img_size,
                "weights": args.weights,
                "metrics": results,
                "eval_duration_sec": eval_duration,
            },
            f,
            indent=2,
        )

    txt_path = os.path.join(args.output_dir, "metrics.txt")
    with open(txt_path, "w") as f:
        f.write(summary_str + "\n")

    print(f"Saved evaluation metrics to: {json_path} and {txt_path}")

    # Automatically update multi-model comparison tables if inside a benchmark results root
    parent_dir = os.path.dirname(os.path.abspath(args.output_dir))
    parent_name = os.path.basename(parent_dir).lower()
    if parent_name.startswith("eval_results"):
        try:
            from summarize_benchmark import generate_comparison_tables

            txt_table, _ = generate_comparison_tables(parent_dir, target_dataset=args.dataset)
            if txt_table:
                print("\n" + txt_table)
        except Exception as e:
            print(f"[Notice] Could not update multi-model comparison table: {e}")

    return results


if __name__ == "__main__":
    main()
