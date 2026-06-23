"""Aggregate metrics across conditions and tiers."""

from __future__ import annotations

from sklearn.metrics import roc_auc_score


def auroc(y_true: list[int], y_score: list[float]) -> float:
    """Compute AUROC from binary labels and continuous scores."""
    return float(roc_auc_score(y_true, y_score))


def confusion_at_threshold(
    y_true: list[int],
    y_score: list[float],
    threshold: float = 0.4,
) -> dict[str, int]:
    """Confusion matrix counts at a score threshold."""
    tp = fp = tn = fn = 0
    for label, score in zip(y_true, y_score, strict=True):
        pred = score >= threshold
        if label == 1 and pred:
            tp += 1
        elif label == 0 and pred:
            fp += 1
        elif label == 0 and not pred:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
