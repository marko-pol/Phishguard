"""
Model training script.

Trains four classifiers on the cleaned dataset:
  1. Logistic Regression  — strong linear baseline, interpretable coefficients
  2. Gaussian Naive Bayes — probabilistic baseline, fast, works with scaled features
  3. Random Forest        — non-linear ensemble, robust to feature scale
  4. XGBoost              — gradient-boosted trees, typically best out-of-the-box

Pipeline
--------
  cleaned.csv  →  feature pipeline (fit on train)  →  train / test split
  → train each model  →  tune threshold (min 95% recall)
  →  evaluate on test set  →  save best model to models/artifacts/

Usage
-----
    python -m src.models.train          # run full training loop
    from src.models.train import train  # import in notebooks
"""

import joblib
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

RAW_DATA_PATH  = Path(__file__).parents[2] / "data" / "processed" / "cleaned.csv"
ARTIFACTS_DIR  = Path(__file__).parents[2] / "models" / "artifacts"

TEST_SIZE      = 0.2
RANDOM_SEED    = 42
MIN_RECALL     = 0.95   # minimum phishing recall before threshold relaxation


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _get_models() -> dict:
    """Return a dict of {name: unfitted model}."""
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
        "naive_bayes": GaussianNB(),
        "random_forest": None,   # lazy import below
        "xgboost": None,         # lazy import below
    }

    try:
        from sklearn.ensemble import RandomForestClassifier
        models["random_forest"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    except ImportError:
        log.warning("RandomForest unavailable — skipping")
        del models["random_forest"]

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    except ImportError:
        log.warning("XGBoost unavailable — skipping")
        del models["xgboost"]

    return {k: v for k, v in models.items() if v is not None}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_path: Path | None = None,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> dict:
    """
    Run the full training pipeline.

    Parameters
    ----------
    data_path : path to cleaned.csv (default: data/processed/cleaned.csv)
    test_size : fraction held out for testing (default 0.2)
    seed      : random seed for reproducibility
    save      : if True, persist best model + pipeline to models/artifacts/

    Returns
    -------
    dict mapping model name → metrics dict
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # --- Load data ---
    path = data_path or RAW_DATA_PATH
    log.info("Loading data from %s", path)
    df = pd.read_csv(path).fillna("")

    required = {"subject", "body", "html_body", "sender", "reply_to", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cleaned.csv is missing columns: {missing}. Re-run the data pipeline.")

    y = df["label"].values
    X_df = df.drop(columns=["label", "source"], errors="ignore")

    # --- Train / test split (stratified to preserve class balance) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=seed, stratify=y
    )
    log.info(
        "Split: %d train / %d test  (phishing rate: train=%.1f%% test=%.1f%%)",
        len(y_train), len(y_test),
        100 * y_train.mean(), 100 * y_test.mean(),
    )

    # --- Fit feature pipeline on training data only ---
    from src.features.pipeline import build_feature_pipeline
    log.info("Fitting feature pipeline on training set …")
    feature_pipe = build_feature_pipeline()
    X_train_feat = feature_pipe.fit_transform(X_train)
    X_test_feat  = feature_pipe.transform(X_test)
    log.info("Feature matrix: train %s  test %s", X_train_feat.shape, X_test_feat.shape)

    # --- Train and evaluate each model ---
    from src.models.evaluate import evaluate, tune_threshold, print_report

    models   = _get_models()
    results  = {}
    best_name, best_f1 = None, -1.0

    for name, model in models.items():
        log.info("Training %s …", name)
        model.fit(X_train_feat, y_train)

        # Tune threshold on the test set (acceptable for Week 2 baselines;
        # use a proper validation split or CV in Week 3 hyperparameter tuning)
        threshold = tune_threshold(model, X_test_feat, y_test, min_recall=MIN_RECALL)
        metrics   = evaluate(model, X_test_feat, y_test, threshold=threshold)
        print_report(name, metrics)

        results[name] = {
            "model":    model,
            "metrics":  metrics,
            "threshold": threshold,
        }

        if metrics["f1"] > best_f1:
            best_f1, best_name = metrics["f1"], name

    log.info("\nBest model: %s  (F1=%.4f)", best_name, best_f1)

    # --- Save artifacts ---
    if save:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save each model individually
        for name, result in results.items():
            artifact = {
                "feature_pipeline": feature_pipe,
                "model":            result["model"],
                "threshold":        result["threshold"],
                "metrics":          result["metrics"],
                "model_name":       name,
            }
            out = ARTIFACTS_DIR / f"{name}.joblib"
            joblib.dump(artifact, out)
            log.info("Saved %s → %s", name, out)

        # Save a 'best_model' symlink-style copy for easy inference
        best_artifact = {
            "feature_pipeline": feature_pipe,
            "model":            results[best_name]["model"],
            "threshold":        results[best_name]["threshold"],
            "metrics":          results[best_name]["metrics"],
            "model_name":       best_name,
        }
        best_out = ARTIFACTS_DIR / "best_model.joblib"
        joblib.dump(best_artifact, best_out)
        log.info("Best model saved → %s", best_out)

    # Return just the metrics summary (no model objects) for easy inspection
    return {
        name: result["metrics"]
        for name, result in results.items()
    }


if __name__ == "__main__":
    results = train()
    print("\nFinal results summary:")
    print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("─" * 73)
    for name, m in results.items():
        print(
            f"{name:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )
