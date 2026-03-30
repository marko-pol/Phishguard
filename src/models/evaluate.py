"""
Model evaluation utilities.

Provides:
  - evaluate()        : compute all classification metrics at a given threshold
  - tune_threshold()  : find the threshold that maximises precision while
                        keeping recall >= a minimum target (default 0.95).
                        Phishing detection penalises false negatives heavily,
                        so we optimise for high recall first.
  - print_report()    : formatted console summary
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics for a fitted model on a test set.

    Parameters
    ----------
    model     : fitted sklearn-compatible model with predict_proba
    X_test    : feature matrix
    y_test    : true labels (0 = ham, 1 = phishing)
    threshold : decision threshold for positive class

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, threshold,
                    confusion_matrix (2x2 list)
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc_auc_score(y_test, y_proba), 4),
        "threshold":        threshold,
        "confusion_matrix": cm.tolist(),
    }


def tune_threshold(
    model,
    X_val,
    y_val,
    min_recall: float = 0.95,
) -> float:
    """
    Find the highest decision threshold (lowest sensitivity) that still
    achieves at least `min_recall` on the positive (phishing) class.

    A higher threshold means fewer false positives; we scan downward from 0.9
    and return the largest threshold where recall >= min_recall.  Falls back
    to 0.5 if no such threshold exists.

    Parameters
    ----------
    model      : fitted model with predict_proba
    X_val      : validation feature matrix
    y_val      : true labels
    min_recall : minimum acceptable recall on class 1 (default 0.95)
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.round(np.arange(0.05, 0.95, 0.01), 2)

    best_threshold = 0.5
    best_precision = 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec  = recall_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        if rec >= min_recall and prec > best_precision:
            best_precision = prec
            best_threshold = float(t)

    return best_threshold


def print_report(model_name: str, metrics: dict) -> None:
    """Print a formatted metrics summary to stdout."""
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"\n{'─' * 46}")
    print(f"  {model_name}")
    print(f"{'─' * 46}")
    print(f"  Threshold  : {metrics['threshold']:.2f}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Confusion matrix:")
    print(f"               Pred Ham  Pred Phish")
    print(f"    True Ham   {tn:>8}  {fp:>10}")
    print(f"    True Phish {fn:>8}  {tp:>10}")
    print(f"{'─' * 46}")
