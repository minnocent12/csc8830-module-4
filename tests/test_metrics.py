from __future__ import annotations

import numpy as np
import pytest

from module4.metrics import confusion_counts, dice, evaluate_masks, iou, precision, recall
from module4.visualization import mask_overlap_bgr


def test_metrics_report_counts_and_scores_for_partial_overlap() -> None:
    prediction = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    reference = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)

    result = evaluate_masks(prediction, reference)

    assert result.tp == 1
    assert result.fp == 1
    assert result.fn == 1
    assert result.tn == 3
    assert result.prediction_foreground_pixels == 2
    assert result.reference_foreground_pixels == 2
    assert result.iou == pytest.approx(1 / 3)
    assert result.dice == pytest.approx(1 / 2)
    assert result.precision == pytest.approx(1 / 2)
    assert result.recall == pytest.approx(1 / 2)


def test_metric_wrappers_share_the_same_pixel_conventions() -> None:
    prediction = np.array([[False, True], [False, False]])
    reference = np.array([[False, True], [False, False]])

    assert confusion_counts(prediction, reference).tp == 1
    assert iou(prediction, reference) == 1.0
    assert dice(prediction, reference) == 1.0
    assert precision(prediction, reference) == 1.0
    assert recall(prediction, reference) == 1.0


def test_empty_foregrounds_are_perfect_only_when_both_masks_are_empty() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    foreground = np.array([[True, False], [False, False]])

    both_empty = evaluate_masks(empty, empty)
    missed = evaluate_masks(empty, foreground)
    false_positive = evaluate_masks(foreground, empty)

    assert (both_empty.iou, both_empty.dice, both_empty.precision, both_empty.recall) == (1.0, 1.0, 1.0, 1.0)
    assert (missed.iou, missed.dice, missed.precision, missed.recall) == (0.0, 0.0, 0.0, 0.0)
    assert (false_positive.iou, false_positive.dice, false_positive.precision, false_positive.recall) == (0.0, 0.0, 0.0, 0.0)


def test_metrics_accept_only_explicit_binary_uint8_encodings() -> None:
    result = evaluate_masks(
        np.array([[0, 255]], dtype=np.uint8),
        np.array([[0, 1]], dtype=np.uint8),
    )
    assert result.tp == 1

    with pytest.raises(ValueError, match="limited to 0, 1, and 255"):
        evaluate_masks(np.array([[0, 128]], dtype=np.uint8), np.zeros((1, 2), dtype=bool))


def test_metrics_reject_shape_mismatch_and_do_not_mutate_inputs() -> None:
    prediction = np.array([[0, 1]], dtype=np.uint8)
    reference = np.array([[0], [1]], dtype=np.uint8)
    original = prediction.copy()

    with pytest.raises(ValueError, match="same height and width"):
        evaluate_masks(prediction, reference)
    np.testing.assert_array_equal(prediction, original)


def test_overlap_visualization_uses_distinct_tp_fp_fn_bgr_colors() -> None:
    prediction = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    reference = np.array([[1, 0, 1, 0]], dtype=np.uint8)

    overlap = mask_overlap_bgr(prediction, reference)

    np.testing.assert_array_equal(
        overlap[0],
        [[0, 200, 0], [0, 0, 255], [255, 0, 0], [0, 0, 0]],
    )
