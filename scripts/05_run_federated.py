"""Stage 7: federated training of the ensemble track (§3.4, §3.5.1).

Run from the project root:
    python scripts/05_run_federated.py

Trains XGBoost and Random Forest across the simulated hospital nodes under
three partition schemes, and evaluates each aggregated global model on the
same held-out test set used by the centralised baseline — so the federation
cost (§3.8.3) is directly measurable.

The three schemes isolate different effects:
  iid            — equal client sizes, similar distributions (idealised)
  non_iid_equal  — equal client sizes, SKEWED distributions
                   (isolates the heterogeneity effect)
  non_iid        — skewed sizes AND distributions (realistic messy case)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.evaluation.metrics import compute_metrics, format_metrics
from ckd_fedxai.federated.aggregation import weighted_average_predictions
from ckd_fedxai.federated.simulation import load_clients, train_federated
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    target = config["data"]["target_column"]
    partitions_dir = ROOT / config["paths"]["partitions_dir"]

    # --- global held-out test set (identical to the centralised baseline's) ---
    test_path = partitions_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"{test_path} not found. Run scripts/04_partition.py first."
        )
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    print(f"Held-out test set: {len(test_df)} patients "
          f"({int((y_test == 1).sum())} CKD / {int((y_test == 0).sum())} non-CKD)\n")

    # --- load centralised baseline for the federation-cost comparison ---
    baseline_path = ROOT / config["paths"]["metrics_dir"] / "central_baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    if not baseline:
        print("! No centralised baseline found. Run 02_train_central.py "
              "to enable federation-cost comparison.\n")

    results: dict = {}

    for scheme in ["iid", "non_iid_equal", "non_iid"]:
        print("=" * 62)
        print(f"PARTITION SCHEME: {scheme.upper()}")
        print("=" * 62)

        try:
            X_clients, y_clients, sizes = load_clients(partitions_dir, scheme, target)
        except FileNotFoundError as err:
            print(f"  ! {err}\n")
            continue

        print(f"  {len(X_clients)} hospital clients, sizes: {sizes}\n")
        results[scheme] = {}

        for model_name in ["random_forest", "xgboost"]:
            print(f"  MODEL: {model_name}")

            # each client trains locally; server aggregates (§3.4.1, §3.4.3)
            client_models = train_federated(
                model_name, X_clients, y_clients, sizes, config
            )

            # size-weighted aggregation of client predictions (Eq. 3.1 analogue)
            y_proba = weighted_average_predictions(client_models, X_test, sizes)
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = compute_metrics(y_test, y_pred, y_proba)

            print("  Federated global model — held-out test metrics:")
            print(format_metrics(metrics))

            # --- federation cost vs centralised baseline (§3.8.3) ---
            if model_name in baseline:
                central_acc = baseline[model_name]["test_metrics"]["accuracy"]
                cost = central_acc - metrics["accuracy"]
                print(f"  Centralised baseline accuracy: {central_acc:.4f}")
                print(f"  FEDERATION COST (accuracy drop): {cost:+.4f}")

            results[scheme][model_name] = metrics
            print()

    # --- summary table across all schemes ---
    if results:
        print("=" * 62)
        print("SUMMARY — accuracy by scheme and model")
        print("=" * 62)
        print(f"{'Scheme':<18}{'Random Forest':<18}{'XGBoost':<18}")
        print("-" * 54)
        if baseline:
            rf_base = baseline.get("random_forest", {}).get("test_metrics", {}).get("accuracy")
            xgb_base = baseline.get("xgboost", {}).get("test_metrics", {}).get("accuracy")
            rf_s = f"{rf_base:.4f}" if rf_base is not None else "—"
            xgb_s = f"{xgb_base:.4f}" if xgb_base is not None else "—"
            print(f"{'centralised':<18}{rf_s:<18}{xgb_s:<18}")
        for scheme, models in results.items():
            rf = models.get("random_forest", {}).get("accuracy")
            xgb = models.get("xgboost", {}).get("accuracy")
            rf_s = f"{rf:.4f}" if rf is not None else "—"
            xgb_s = f"{xgb:.4f}" if xgb is not None else "—"
            print(f"{scheme:<18}{rf_s:<18}{xgb_s:<18}")
        print("-" * 54)

    # --- save ---
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "federated_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved federated results to: {out}")


if __name__ == "__main__":
    main()