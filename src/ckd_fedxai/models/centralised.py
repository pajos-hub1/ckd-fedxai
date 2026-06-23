"""Centralised baseline models (§3.5.3).

Trains XGBoost, Random Forest, and Logistic Regression on pooled data.
This is the non-federated, no-privacy reference against which the
federation cost and privacy cost are later measured (§3.8.3).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_models(config: dict) -> dict:
    """Instantiate the three baseline models from config hyperparameters."""
    rf_cfg = config["models"]["random_forest"]
    xgb_cfg = config["models"]["xgboost"]
    lr_cfg = config["models"]["logistic_regression"]

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"],
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            random_state=xgb_cfg["random_state"],
            eval_metric="logloss",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=lr_cfg["max_iter"],
            random_state=lr_cfg["random_state"],
        ),
    }
    return models