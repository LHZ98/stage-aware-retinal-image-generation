"""
Evaluation metrics: Quadratic Weighted Kappa, Accuracy, Macro F1, confusion matrix.
"""
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa for ordered 0-4 classification."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def one_off_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions within one grade of the true label."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return QWK, Accuracy, Macro F1, One-off accuracy, confusion matrix."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "one_off_accuracy": one_off_accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Print metrics and confusion matrix."""
    print(f"{prefix}QWK: {metrics['qwk']:.4f}")
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}Macro F1: {metrics['macro_f1']:.4f}")
    print(f"{prefix}One-off Accuracy: {metrics['one_off_accuracy']:.4f}")
    print(f"{prefix}Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"{prefix}  {row}")
