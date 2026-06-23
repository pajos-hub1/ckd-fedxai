"""Stage 4: train and evaluate the centralised baseline models (§3.5.3).

Run from the project root:
    python scripts/02_train_central.py

Strategy (§3.9.3): hold out a test set the models never see; run
stratified k-fold CV on the training portion to show stability; then
report final metrics on the held-out test set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.evaluation.metrics import compute_metrics, format_metrics
from ckd_fedxai.models.centralised import build_models
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    # --- load processed data ---
    data_path = ROOT / config["paths"]["processed_data"]
    df = pd.read_csv(data_path)
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    # --- held-out test set (never seen during CV/training) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        stratify=y,
        random_state=config["seed"],
    )
    print(f"Train: {X_train.shape[0]} rows | Test (held-out): {X_test.shape[0]} rows")

    models = build_models(config)
    cv = StratifiedKFold(
        n_splits=config["evaluation"]["cv_folds"],
        shuffle=True,
        random_state=config["seed"],
    )

    all_results = {}
    models_dir = ROOT / config["paths"]["models_dir"]
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print("\n" + "=" * 55)
        print(f"MODEL: {name}")
        print("=" * 55)

        # --- cross-validation on the training set (robustness) ---
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"CV accuracy ({config['evaluation']['cv_folds']}-fold): "
              f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # --- fit on full training set, evaluate on held-out test ---
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)

        print("Held-out test metrics:")
        print(format_metrics(test_metrics))

        # --- save model + metrics ---
        import joblib
        joblib.dump(model, models_dir / f"central_{name}.joblib")

        all_results[name] = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_metrics": test_metrics,
        }

    # --- save consolidated metrics ---
    out = metrics_dir / "central_baseline.json"
    with open(out, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n✓ Saved baseline metrics to: {out}")
    print(f"✓ Saved trained models to: {models_dir}")


if __name__ == "__main__":
    main()