"""Unified Multi-Model Multi-Dataset Training Driver for Scene Change Detection (SCD).

Supports:
1. Models:
   - 'paradis_scd': ParaDis-SCD (CVPR Main Conference Model, optimized via ParaDisLoss).
   - 'csnet': ChangeSim Baseline (ICRA 2021).
   - 'changeformer': Transformer Baseline (IEEE JSTARS 2022).
   - 'dr_tanet': Dual-Resolution Spatio-Temporal Attention Baseline (IEEE TIP 2020).
2. Datasets:
   - 'changesim', 'vlcmu', 'pcd' (with seamless synthetic offline fallbacks).
3. Mixed Precision (FP16 AMP) training with PyTorch GradScaler.
4. SCDMetrics cumulative evaluation tracking 7 key metrics (F1, mIoU, OA, etc.).
5. Checkpoint management saving 'latest.pth' and 'best.pth'.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure cvpr2 is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paradis_scd.models import ParaDisSCDNet
from paradis_scd.losses import ParaDisLoss, FocalLoss, SoftDiceLoss
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
    """Factory for loading supported SCD datasets."""
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
    """Factory for instantiating model architecture."""
    name_clean = name.lower().replace("-", "_").replace(" ", "")
    if name_clean in ("paradis_scd", "paradis", "paradisscd"):
        return ParaDisSCDNet(pretrained=False, freeze_priors=False, ablation=ablation)
    else:
        return get_baseline_model(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SCD Training Driver")
    parser.add_argument(
        "--model",
        type=str,
        default="paradis_scd",
        choices=SUPPORTED_MODELS,
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="changesim",
        choices=["changesim", "vlcmu", "pcd"],
        help="Dataset benchmark",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img_size", type=int, nargs=2, default=[480, 640], help="Image height and width")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory for dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable FP16 Automatic Mixed Precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 1 train and 1 val step for smoke testing and verification",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_dbaia", "no_desg", "no_ecdca", "no_pdag"],
        help="Ablation mode for ParaDis-SCD",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    criterion: nn.Module,
    is_paradis: bool,
    focal_loss: Optional[nn.Module],
    dice_loss: Optional[nn.Module],
    device: torch.device,
    use_amp: bool,
    epoch: int,
    dry_run: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(loader):
        img1 = batch["img1"].to(device, non_blocking=True)
        img2 = batch["img2"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            outputs = model(img1, img2)
            if is_paradis:
                loss_dict = criterion(outputs, mask)
                loss = loss_dict["loss"]
            else:
                logits = outputs["logits"]
                l_focal = focal_loss(logits, mask)
                l_dice = dice_loss(logits, mask)
                loss = l_focal + l_dice
                if "aux_logits" in outputs and outputs["aux_logits"] is not None:
                    aux_logits = outputs["aux_logits"]
                    loss = loss + 0.5 * (focal_loss(aux_logits, mask) + dice_loss(aux_logits, mask))

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 10 == 0 or dry_run:
            print(f"  [Epoch {epoch}][Step {step+1}/{len(loader)}] Loss: {loss.item():.4f}")

        if dry_run:
            break

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dry_run: bool = False,
) -> Dict[str, float]:
    model.eval()
    metrics = SCDMetrics()

    for step, batch in enumerate(loader):
        img1 = batch["img1"].to(device, non_blocking=True)
        img2 = batch["img2"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        outputs = model(img1, img2)
        logits = outputs["logits"]
        metrics.update(logits, mask)

        if dry_run:
            break

    return metrics.compute()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    target_img_size = (args.img_size[0], args.img_size[1])

    print("=" * 70)
    print("SCD UNIFIED TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Resolution  : {target_img_size[0]}x{target_img_size[1]}")
    print(f"  Device      : {device}")
    print(f"  AMP Enabled : {args.amp}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Save Dir    : {args.save_dir}")
    print(f"  Dry Run     : {args.dry_run}")
    print("-" * 70)

    # 1. Datasets & Loaders
    train_dataset = get_dataset(
        name=args.dataset,
        split="train",
        root_dir=args.data_dir,
        img_size=target_img_size,
    )
    val_dataset = get_dataset(
        name=args.dataset,
        split="val",
        root_dir=args.data_dir,
        img_size=target_img_size,
    )
    print(f"  Train Data  : {len(train_dataset)} samples (is_synthetic={train_dataset.is_synthetic}, root={train_dataset.root_dir})")
    print(f"  Val Data    : {len(val_dataset)} samples (is_synthetic={val_dataset.is_synthetic}, root={val_dataset.root_dir})")
    print("-" * 70)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) > args.batch_size and args.batch_size > 1),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model & Optimizer
    model = get_model(args.model, ablation=args.ablation).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if (args.amp and device.type == "cuda") else None

    is_paradis = args.model.lower().replace("-", "_") in ("paradis_scd", "paradis")
    paradis_criterion = ParaDisLoss() if is_paradis else None
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0) if not is_paradis else None
    dice_loss = SoftDiceLoss(smooth=1.0) if not is_paradis else None

    # Checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = -1.0
    start_epoch = 1

    # Resume from existing checkpoint if requested
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_f1 = checkpoint.get("best_f1", -1.0)
        print(f"Resumed from epoch {start_epoch-1} with best F1: {best_f1:.4f}")

    total_epochs = 1 if args.dry_run else args.epochs

    # Training Loop
    t_start = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=paradis_criterion,
            is_paradis=is_paradis,
            focal_loss=focal_loss,
            dice_loss=dice_loss,
            device=device,
            use_amp=args.amp,
            epoch=epoch,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            scheduler.step()

        val_metrics = {}
        if epoch % args.val_interval == 0 or epoch == total_epochs or args.dry_run:
            val_metrics = validate(model=model, loader=val_loader, device=device, dry_run=args.dry_run)
            f1 = val_metrics["f1"]
            miou = val_metrics["miou"]
            oa = val_metrics["oa"]
            print(
                f"[Epoch {epoch}/{total_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {f1*100:.2f}% | "
                f"mIoU: {miou*100:.2f}% | "
                f"OA: {oa*100:.2f}% "
                f"({time.time() - epoch_start:.1f}s)"
            )

            # Checkpoint metadata
            checkpoint_data = {
                "epoch": epoch,
                "model_name": args.model,
                "dataset": args.dataset,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_f1": best_f1,
                "metrics": val_metrics,
                "args": vars(args),
            }

            # Save latest checkpoint
            latest_path = os.path.join(args.save_dir, "latest.pth")
            torch.save(checkpoint_data, latest_path)

            # Save best checkpoint
            if f1 > best_f1:
                best_f1 = f1
                checkpoint_data["best_f1"] = best_f1
                best_path = os.path.join(args.save_dir, "best.pth")
                torch.save(checkpoint_data, best_path)
                print(f"  >>> New Best F1: {best_f1*100:.2f}%! Saved {best_path}")

        if args.dry_run:
            print("  [Dry Run Complete] Executed 1 train step and 1 val step successfully.")
            break

    print("=" * 70)
    print(f"TRAINING COMPLETE in {time.time() - t_start:.2f}s! Best F1: {best_f1*100:.2f}%")
    print(f"Checkpoints preserved in: {os.path.abspath(args.save_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
