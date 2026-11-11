"""Fixed-label classification metrics; no labels enter frozen extraction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)


def classification_metrics(probabilities: np.ndarray, truth: np.ndarray,
                           classes: list[str]) -> dict:
    predicted = probabilities.argmax(1)
    count = len(classes)
    top_k = min(3, count)
    top = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(range(count)), zero_division=0,
    )
    return {
        "samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=list(range(count)),
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=list(range(count)),
                                      average="weighted", zero_division=0)),
        "top_3_accuracy": float(np.mean(np.any(top == truth[:, None], axis=1))),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(count))).tolist(),
        "class_order": classes,
        "per_class": {
            name: {"precision": float(precision[index]), "recall": float(recall[index]),
                   "f1": float(f1[index]), "support": int(support[index])}
            for index, name in enumerate(classes)
        },
    }
