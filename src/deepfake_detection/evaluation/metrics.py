from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


class ClassificationMetrics:
    def __init__(self, f1: float, ap: float, auc: float):
        self.f1 = f1
        self.ap = ap
        self.auc = auc

    def as_dict(self) -> dict[str, float]:
        return {"f1": self.f1, "ap": self.ap, "auc": self.auc}


def compute_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    predictions = (probs >= threshold).astype(int)
    f1 = f1_score(targets, predictions, zero_division=0)
    ap = average_precision_score(targets, probs)
    auc = roc_auc_score(targets, probs)
    return ClassificationMetrics(f1=f1, ap=ap, auc=auc)
