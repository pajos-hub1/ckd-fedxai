"""Stage 7b: multi-seed federated experiments (§3.4.2, §3.9.3).

Run from the project root:
    python scripts/05b_federated_multiseed.py

Repeats the full partition -> local training -> aggregation -> evaluation
cycle across N random seeds, and reports mean ± std for each partition
scheme. This distinguishes a genuine heterogeneity effect from the random
variation inherent in partitioning a small dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.data.partition import partition
from ckd_fedxai.evaluation.metrics import compute_metrics
from ckd_fedxai.federated.aggregation import weighted_average_predictions
from ckd_fedxai.models.centralised import build_models
from ckd_fedxai.utils.config import load_config


SCHEMES = ["iid", "non_iid_equal", "non_iid"]
MODELS = ["random_forest", "xgboost"]


def run_one_seed(df: pd.DataFrame, config: dict, seed: int) -> dict:
    """One complete experiment at a given seed: split, partition, train, evaluate."""
    target = config["data"]["target_column"]

    # fresh train/test split for THIS seed
    train_df, test_df = train_test_split(
        df, test_size=config["data"]["test_size"],
        stratify=df[target], random_state=seed,
    )
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # seed-specific config so partitioning and models both vary
    cfg = json.loads(json.dumps(config))  # deep copy
    cfg["seed"] = seed
    for m in MODELS:
        cfg["models"][m]["random_state"] = seed

    results: dict = {"centralised": {}, **{s: {} for s in SCHEMES}}

    # --- centralised baseline at this seed ---
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    for model_name in MODELS:
        model = build_models(cfg)[model_name]
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results["centralised"][model_name] = compute_metrics(y_test, pred, proba)

    # --- federated, per scheme ---
    for scheme in SCHEMES:
        parts = partition(train_df, cfg, scheme=scheme)
        sizes = [len(p) for p in parts]

        for model_name in MODELS:
            client_models = []
            for part in parts:
                Xc = part.drop(columns=[target])
                yc = part[target]
                local = build_models(cfg)[model_name]
                local.fit(Xc, yc)
                client_models.append(local)

            proba = weighted_average_predictions(client_models, X_test, sizes)
            pred = (proba >= 0.5).astype(int)
            results[scheme][model_name] = compute_metrics(y_test, pred, proba)

    return results


def main() -> None:
    config = load_config()
    n_seeds = config["federated"].get("num_seeds", 10)
    base_seed = config["seed"]

    df = pd.read_csv(ROOT / config["paths"]["processed_data"])

    seeds = [base_seed + i for i in range(n_seeds)]
    print(f"Running {n_seeds} seeds: {seeds}\n")

    # collect: all_runs[scheme][model] = list of metric dicts
    all_runs: dict = {"centralised": {m: [] for m in MODELS}}
    for s in SCHEMES:
        all_runs[s] = {m: [] for m in MODELS}

    for i, seed in enumerate(seeds, 1):
        print(f"  seed {seed}  ({i}/{n_seeds}) ...", end=" ", flush=True)
        res = run_one_seed(df, config, seed)
        for scheme in all_runs:
            for model_name in MODELS:
                all_runs[scheme][model_name].append(res[scheme][model_name])
        print("done")

    # --- aggregate to mean ± std ---
    print("\n" + "=" * 78)
    print(f"RESULTS ACROSS {n_seeds} SEEDS  (mean ± std)")
    print("=" * 78)

    summary: dict = {}
    for model_name in MODELS:
        print(f"\nMODEL: {model_name}")
        print(f"{'Scheme':<18}{'Accuracy':<20}{'Recall':<20}{'AUC-ROC':<20}")
        print("-" * 78)
        summary[model_name] = {}

        for scheme in ["centralised", *SCHEMES]:
            runs = all_runs[scheme][model_name]
            acc = np.array([r["accuracy"] for r in runs])
            rec = np.array([r["recall"] for r in runs])
            auc = np.array([r["auc_roc"] for r in runs])

            print(f"{scheme:<18}"
                  f"{acc.mean():.4f} ± {acc.std():.4f}   "
                  f"{rec.mean():.4f} ± {rec.std():.4f}   "
                  f"{auc.mean():.4f} ± {auc.std():.4f}")

            summary[model_name][scheme] = {
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std()),
                "recall_mean": float(rec.mean()),
                "recall_std": float(rec.std()),
                "auc_mean": float(auc.mean()),
                "auc_std": float(auc.std()),
            }

        # --- federation cost, averaged ---
        central = summary[model_name]["centralised"]["accuracy_mean"]
        print("-" * 78)
        for scheme in SCHEMES:
            fed = summary[model_name][scheme]["accuracy_mean"]
            print(f"  federation cost ({scheme:<14}): {central - fed:+.4f}")

    out = ROOT / config["paths"]["metrics_dir"] / "federated_multiseed.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Saved multi-seed results to: {out}")


if __name__ == "__main__":
    main()