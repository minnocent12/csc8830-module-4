"""Pixel-level segmentation metrics for aligned canonical binary masks."""
from __future__ import annotations

import numpy as np

from module4.types import ConfusionCounts, SegmentationMetrics
from module4.validation import validate_mask_pair


def confusion_counts(prediction: np.ndarray, reference: np.ndarray) -> ConfusionCounts:
    """Return integer TP, FP, FN, and TN counts using boolean operations."""
    prediction_bool, reference_bool = validate_mask_pair(prediction, reference)
    return ConfusionCounts(
        tp=int(np.count_nonzero(prediction_bool & reference_bool)),
        fp=int(np.count_nonzero(prediction_bool & ~reference_bool)),
        fn=int(np.count_nonzero(~prediction_bool & reference_bool)),
        tn=int(np.count_nonzero(~prediction_bool & ~reference_bool)),
    )


def _metric_values(counts: ConfusionCounts) -> tuple[float, float, float, float]:
    tp, fp, fn = counts.tp, counts.fp, counts.fn
    union = tp + fp + fn
    foreground_total = 2 * tp + fp + fn
    if union == 0:
        iou_value = 1.0
    else:
        iou_value = tp / union
    if foreground_total == 0:
        dice_value = 1.0
    else:
        dice_value = (2 * tp) / foreground_total
    precision_value = 1.0 if tp + fp == 0 and tp + fn == 0 else (tp / (tp + fp) if tp + fp else 0.0)
    recall_value = 1.0 if tp + fp == 0 and tp + fn == 0 else (tp / (tp + fn) if tp + fn else 0.0)
    return float(iou_value), float(dice_value), float(precision_value), float(recall_value)


def evaluate_masks(prediction: np.ndarray, reference: np.ndarray) -> SegmentationMetrics:
    """Compute all pixel-level metrics from one shared confusion-count calculation."""
    prediction_bool, reference_bool = validate_mask_pair(prediction, reference)
    counts = confusion_counts(prediction_bool, reference_bool)
    iou_value, dice_value, precision_value, recall_value = _metric_values(counts)
    return SegmentationMetrics(
        iou=iou_value,
        dice=dice_value,
        precision=precision_value,
        recall=recall_value,
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        tn=counts.tn,
        prediction_foreground_pixels=counts.tp + counts.fp,
        reference_foreground_pixels=counts.tp + counts.fn,
    )


def iou(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return pixel-level intersection over union."""
    return evaluate_masks(prediction, reference).iou


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return pixel-level Dice coefficient."""
    return evaluate_masks(prediction, reference).dice


def precision(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return pixel-level foreground precision."""
    return evaluate_masks(prediction, reference).precision


def recall(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return pixel-level foreground recall."""
    return evaluate_masks(prediction, reference).recall
