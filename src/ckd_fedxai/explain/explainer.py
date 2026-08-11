"""Explainability module: SHAP (global) and LIME (local) — §3.6.

Model-aware: picks the correct SHAP explainer for the model type
(TreeExplainer for XGBoost/RF, LinearExplainer for logistic regression).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, save to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return "forest" in name or "xgb" in name or "boost" in name or "tree" in name


def shap_global(model, X: pd.DataFrame, model_name: str,
                figures_dir: Path, max_display: int = 14) -> pd.DataFrame:
    """Compute SHAP values and save a summary plot. Returns importance ranking."""
    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        # linear / other models
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    # Some explainers return a list (per class); take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # Newer SHAP may return 3D array (n, features, classes)
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # --- summary plot (beeswarm) ---
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    beeswarm_path = figures_dir / f"shap_summary_{model_name}.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- bar plot (mean |SHAP|) ---
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar",
                      max_display=max_display, show=False)
    plt.tight_layout()
    bar_path = figures_dir / f"shap_importance_{model_name}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- numeric importance ranking ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print(f"  Saved SHAP plots: {beeswarm_path.name}, {bar_path.name}")
    return ranking


def lime_local(model, X_train: pd.DataFrame, X_explain: pd.DataFrame,
               model_name: str, figures_dir: Path, num_samples: int = 5) -> None:
    """Generate LIME explanations for a few individual patients."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=["notckd", "ckd"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    n = min(num_samples, len(X_explain))
    for i in range(n):
        row = X_explain.iloc[i].values
        exp = explainer.explain_instance(
            row, model.predict_proba, num_features=10
        )
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        out = figures_dir / f"lime_{model_name}_patient{i+1}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {n} LIME patient explanations.")


def lime_fidelity_and_stability(model, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                                num_samples: int = 5, num_repeats: int = 5,
                                top_k: int = 5) -> pd.DataFrame:
    """Quantify LIME's fidelity and stability (§3.8.2), on the same patients
    explained by lime_local().

    Fidelity: the R^2 of LIME's local linear surrogate against the black-box
    model's predictions in the perturbed neighbourhood around each patient
    (LIME's own `exp.score`), at the canonical seed (42) used for the saved
    patient explanations in lime_local().

    Stability: LIME is repeated `num_repeats` times per patient, each with an
    independent random seed, and the Jaccard similarity of the top-`top_k`
    feature set (by absolute weight, for the "ckd" class) is measured between
    every pair of repeats. A low mean Jaccard similarity means the features
    LIME reports as most influential change from run to run for the same
    patient -- the instability noted in Section 2.6.2 of the literature review.
    """
    from itertools import combinations

    from lime.lime_tabular import LimeTabularExplainer

    ckd_label = list(model.classes_).index(1)
    n = min(num_samples, len(X_explain))
    rows = []

    for i in range(n):
        row = X_explain.iloc[i].values

        canonical = LimeTabularExplainer(
            training_data=X_train.values, feature_names=list(X_train.columns),
            class_names=["notckd", "ckd"], mode="classification",
            discretize_continuous=True, random_state=42,
        )
        exp = canonical.explain_instance(row, model.predict_proba, num_features=10)
        fidelity = float(exp.score)

        top_sets = []
        for r in range(num_repeats):
            rep_explainer = LimeTabularExplainer(
                training_data=X_train.values, feature_names=list(X_train.columns),
                class_names=["notckd", "ckd"], mode="classification",
                discretize_continuous=True, random_state=100 + r,
            )
            rep_exp = rep_explainer.explain_instance(
                row, model.predict_proba, num_features=10
            )
            ranked = sorted(rep_exp.as_map()[ckd_label], key=lambda t: -abs(t[1]))
            top_sets.append({idx for idx, _ in ranked[:top_k]})

        pairwise_jaccard = [
            len(a & b) / len(a | b) for a, b in combinations(top_sets, 2)
        ]
        stability = float(np.mean(pairwise_jaccard))

        rows.append({
            "patient": i + 1,
            "fidelity_r2": fidelity,
            "stability_jaccard": stability,
        })

    return pd.DataFrame(rows)