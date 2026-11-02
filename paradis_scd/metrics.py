"""ParaDis-SCD Evaluation Metrics: Streaming Cumulative Confusion Matrix.

Implements streaming evaluation for binary scene change detection (SCD):
- Precision = TP / (TP + FP + eps)
- Recall = TP / (TP + FN + eps)
- F1 = 2 * TP / (2 * TP + FP + FN + eps)
- Change_IoU = TP / (TP + FP + FN + eps)
- Static_IoU = TN / (TN + FP + FN + eps)
- mIoU = (Change_IoU + Static_IoU) / 2.0
- Overall Accuracy (OA) = (TP + TN) / (TP + TN + FP + FN + eps)
"""

from typing import Dict, Optional, Union
import numpy as np
import torch


class SCDMetrics:
    """Streaming Cumulative Confusion Matrix for Scene Change Detection.

    Args:
        threshold: Decision threshold for continuous change predictions (default 0.5).
        eps: Small epsilon to prevent zero division in empty classes (default 1e-7).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-7,
    ):
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        """Resets all cumulative confusion matrix counts to zero."""
        self.tp: int = 0
        self.fp: int = 0
        self.tn: int = 0
        self.fn: int = 0
        self.total_pixels: int = 0

    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Updates streaming counts with batch predictions and ground truth targets.

        Args:
            preds: Predicted change logits, probabilities, or binary masks [B, ...].
            targets: Ground truth binary change masks [B, ...] in {0, 1}.
        """
        # Convert to numpy / torch boolean arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()
            if preds.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                # If values are raw logits with unbounded range, apply sigmoid
                if preds.min() < 0.0 or preds.max() > 1.0:
                    preds = torch.sigmoid(preds)
                pred_binary = (preds >= self.threshold).numpy().astype(bool).ravel()
            else:
                pred_binary = (preds > 0).numpy().astype(bool).ravel()
        elif isinstance(preds, np.ndarray):
            if np.issubdtype(preds.dtype, np.floating):
                if preds.min() < 0.0 or preds.max() > 1.0:
                    # Apply numerically safe sigmoid
                    preds = 1.0 / (1.0 + np.exp(-np.clip(preds, -50.0, 50.0)))
                pred_binary = (preds >= self.threshold).ravel()
            else:
                pred_binary = (preds > 0).ravel()
        else:
            raise TypeError(f"Unsupported preds type: {type(preds)}")

        if isinstance(targets, torch.Tensor):
            target_binary = (targets.detach().cpu() > 0.5).numpy().astype(bool).ravel()
        elif isinstance(targets, np.ndarray):
            target_binary = (targets > 0.5).ravel()
        else:
            raise TypeError(f"Unsupported targets type: {type(targets)}")

        if pred_binary.shape != target_binary.shape:
            raise ValueError(f"Shape mismatch: preds {pred_binary.shape} vs targets {target_binary.shape}")

        tp = int(np.sum(pred_binary & target_binary))
        fp = int(np.sum(pred_binary & (~target_binary)))
        tn = int(np.sum((~pred_binary) & (~target_binary)))
        fn = int(np.sum((~pred_binary) & target_binary))

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        self.total_pixels += len(pred_binary)

    def compute(self) -> Dict[str, float]:
        """Computes and returns all evaluation metrics from cumulative counts.

        Returns:
            Dictionary containing metrics:
                'precision', 'recall', 'f1', 'change_iou', 'static_iou', 'miou', 'oa'
            along with capitalized aliases:
                'Precision', 'Recall', 'F1', 'Change_IoU', 'Static_IoU', 'mIoU', 'OA'
        """
        tp = float(self.tp)
        fp = float(self.fp)
        tn = float(self.tn)
        fn = float(self.fn)
        eps = self.eps

        # Precision & Recall
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # F1-Score
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0

        # Intersection over Union for Change class
        change_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        # Intersection over Union for Static class
        static_iou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else (1.0 if (self.total_pixels > 0 and tn > 0) else 0.0)

        # Mean IoU
        miou = (change_iou + static_iou) / 2.0

        # Overall Accuracy
        oa = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0


        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'change_iou': float(change_iou),
            'static_iou': float(static_iou),
            'miou': float(miou),
            'oa': float(oa),
            'tp': int(self.tp),
            'fp': int(self.fp),
            'tn': int(self.tn),
            'fn': int(self.fn),
            'total_pixels': int(self.total_pixels),
            # Capitalized aliases
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'Change_IoU': float(change_iou),
            'Static_IoU': float(static_iou),
            'mIoU': float(miou),
            'OA': float(oa),
        }
        return metrics

    def summary(self) -> str:
        """Returns a formatted tabular summary of cumulative metrics."""
        m = self.compute()
        lines = [
            "=" * 50,
            "ParaDis-SCD Cumulative Evaluation Metrics",
            "=" * 50,
            f"  Overall Accuracy (OA) : {m['oa'] * 100:.2f}%",
            f"  Change F1-Score       : {m['f1'] * 100:.2f}%",
            f"  Mean IoU (mIoU)       : {m['miou'] * 100:.2f}%",
            f"  Change IoU            : {m['change_iou'] * 100:.2f}%",
            f"  Static IoU            : {m['static_iou'] * 100:.2f}%",
            f"  Precision             : {m['precision'] * 100:.2f}%",
            f"  Recall                : {m['recall'] * 100:.2f}%",
            "-" * 50,
            f"  TP: {self.tp:,} | FP: {self.fp:,} | TN: {self.tn:,} | FN: {self.fn:,}",
            f"  Total Pixels Evaluated: {self.total_pixels:,}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        m = self.compute()
        return (
            f"SCDMetrics(F1={m['f1']:.4f}, mIoU={m['miou']:.4f}, "
            f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, OA={m['oa']:.4f})"
        )
