"""Stage 8a: differential privacy on the ensemble track (§3.7.1).

Run from the project root:
    python scripts/06_privacy_dp.py

Sweeps the privacy budget epsilon and measures the resulting loss in
predictive performance — producing the privacy-utility trade-off curve
that is a central result of Objective 5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.evaluation.metrics import compute_metrics, format_metrics
from ckd_fedxai.federated.aggregation import weighted_average_predictions
from ckd_fedxai.federated.simulation import load_clients, train_federated
from ckd_fedxai.privacy.differential_privacy import dp_weighted_aggregate
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    target = config["data"]["target_column"]
    partitions_dir = ROOT / config["paths"]["partitions_dir"]

    test_df = pd.read_csv(partitions_dir / "test.csv")
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    dp_cfg = config["privacy"]["differential_privacy"]
    epsilons = dp_cfg["epsilons"]
    delta = dp_cfg["delta"]
    mechanism = dp_cfg["mechanism"]
    scheme = config["federated"]["partition_scheme"]

    print(f"Partition scheme: {scheme}")
    print(f"DP mechanism: {mechanism}, delta = {delta}")
    print(f"Privacy budgets to sweep: {epsilons}\n")

    X_clients, y_clients, sizes = load_clients(partitions_dir, scheme, target)

    results: dict = {}

    for model_name in ["random_forest", "xgboost"]:
        print("=" * 66)
        print(f"MODEL: {model_name}")
        print("=" * 66)

        client_models = train_federated(model_name, X_clients, y_clients,
                                        sizes, config)

        # --- no-privacy reference (epsilon = infinity) ---
        proba = weighted_average_predictions(client_models, X_test, sizes)
        pred = (proba >= 0.5).astype(int)
        base = compute_metrics(y_test, pred, proba)
        print(f"\n  NO PRIVACY (epsilon = inf):")
        print(format_metrics(base))

        results[model_name] = {"no_privacy": base, "dp": {}}

        # --- sweep epsilon ---
        for eps in epsilons:
            proba = dp_weighted_aggregate(
                client_models, X_test, sizes,
                epsilon=eps, delta=delta, mechanism=mechanism,
                seed=config["seed"],
            )
            pred = (proba >= 0.5).astype(int)
            m = compute_metrics(y_test, pred, proba)

            cost = base["accuracy"] - m["accuracy"]
            print(f"\n  epsilon = {eps}:")
            print(format_metrics(m))
            print(f"  PRIVACY COST (accuracy drop): {cost:+.4f}")

            results[model_name]["dp"][str(eps)] = {
                **m, "privacy_cost": float(cost)
            }
        print()

    # --- privacy-utility trade-off plot ---
    figures_dir = ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for model_name in results:
        eps_vals = [float(e) for e in results[model_name]["dp"].keys()]
        accs = [results[model_name]["dp"][str(e)]["accuracy"] for e in eps_vals]
        order = np.argsort(eps_vals)
        plt.plot(np.array(eps_vals)[order], np.array(accs)[order],
                 marker="o", label=model_name)
        plt.axhline(results[model_name]["no_privacy"]["accuracy"],
                    linestyle="--", alpha=0.4)

    plt.xlabel("Privacy budget ε  (smaller = stronger privacy)")
    plt.ylabel("Accuracy")
    plt.title("Privacy–Utility Trade-off (Differential Privacy)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = figures_dir / "privacy_utility_dp.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"✓ Saved privacy–utility plot: {out_fig}")

    out = ROOT / config["paths"]["metrics_dir"] / "privacy_dp.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"✓ Saved DP results to: {out}")


if __name__ == "__main__":
    main()