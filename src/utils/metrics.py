"""Metrics and prediction helpers for multi-label ingredient recognition."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid function to logits."""
    return 1.0 / (1.0 + np.exp(-x))


def multilabel_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute common multi-label metrics from raw model logits.

    We threshold probabilities because every ingredient class is an independent
    yes/no decision.
    """
    probabilities = sigmoid(logits)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "micro_f1": f1_score(targets, predictions, average="micro", zero_division=0),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "precision": precision_score(
            targets, predictions, average="micro", zero_division=0
        ),
        "recall": recall_score(targets, predictions, average="micro", zero_division=0),
    }


def extract_top_k_ingredients(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-k predicted ingredients with probabilities."""
    probabilities_np = np.asarray(probabilities)
    sorted_indices = np.argsort(probabilities_np)[::-1][:top_k]
    return [(class_names[idx], float(probabilities_np[idx])) for idx in sorted_indices]
