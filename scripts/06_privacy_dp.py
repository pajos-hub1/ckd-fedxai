"""Stage 8a: differential privacy on the ensemble track (§3.7.1).

Run from the project root:
    python scripts/06_privacy_dp.py

Sweeps the privacy budget epsilon across N seeds -- fresh stratified
train/test split and fresh partition per seed, mirroring
05b_federated_multiseed.py's and 05c_federated_lr.py's convention -- and
measures the resulting loss in predictive performance, reporting mean ±
std per epsilon. This is the privacy-utility trade-off curve that is a
central result of Objective 5; multi-seeding it means the curve reflects
genuine robustness rather than a single partition draw.
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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.data.partition import partition
from ckd_fedxai.evaluation.metrics import compute_metrics
from ckd_fedxai.federated.aggregation import weighted_average_predictions
from ckd_fedxai.models.centralised import build_models
from ckd_fedxai.privacy.differential_privacy import dp_weighted_aggregate
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed

MODELS = ["random_forest", "xgboost"]


def run_one_seed(df: pd.DataFrame, config: dict, scheme: str, seed: int,
                 epsilons: list, delta: float, mechanism: str) -> dict:
    """One full DP experiment at a given seed: fresh stratified split,
    fresh partition, local training, the no-privacy baseline, and the
    full epsilon sweep.
    """
    target = config["data"]["target_column"]
    train_df, test_df = train_test_split(
        df, test_size=config["data"]["test_size"],
        stratify=df[target], random_state=seed,
    )
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    cfg = json.loads(json.dumps(config))  # deep copy
    cfg["seed"] = seed
    for m in MODELS:
        cfg["models"][m]["random_state"] = seed

    parts = partition(train_df, cfg, scheme=scheme)
    X_clients = [p.drop(columns=[target]) for p in parts]
    y_clients = [p[target] for p in parts]
    sizes = [len(p) for p in parts]

    results: dict = {}
    for model_name in MODELS:
        client_models = []
        for Xc, yc in zip(X_clients, y_clients):
            local = build_models(cfg)[model_name]
            local.fit(Xc, yc)
            client_models.append(local)

        proba = weighted_average_predictions(client_models, X_test, sizes)
        pred = (proba >= 0.5).astype(int)
        base = compute_metrics(y_test, pred, proba)

        dp_results = {}
        for eps in epsilons:
            proba_dp = dp_weighted_aggregate(
                client_models, X_test, sizes,
                epsilon=eps, delta=delta, mechanism=mechanism, seed=seed,
            )
            pred_dp = (proba_dp >= 0.5).astype(int)
            m = compute_metrics(y_test, pred_dp, proba_dp)
            m["privacy_cost"] = float(base["accuracy"] - m["accuracy"])
            dp_results[str(eps)] = m

        results[model_name] = {"no_privacy": base, "dp": dp_results}
    return results


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    target = config["data"]["target_column"]
    dp_cfg = config["privacy"]["differential_privacy"]
    epsilons = dp_cfg["epsilons"]
    delta = dp_cfg["delta"]
    mechanism = dp_cfg["mechanism"]
    scheme = config["federated"]["partition_scheme"]
    n_seeds = config["federated"].get("num_seeds", 10)
    base_seed = config["seed"]
    seeds = [base_seed + i for i in range(n_seeds)]

    print(f"Partition scheme: {scheme}")
    print(f"DP mechanism: {mechanism}, delta = {delta}")
    print(f"Privacy budgets to sweep: {epsilons}")
    print(f"Seeds: {seeds}\n")

    df = pd.read_csv(ROOT / config["paths"]["processed_data"])

    all_runs = {
        m: {"no_privacy": [], "dp": {str(e): [] for e in epsilons}} for m in MODELS
    }

    for i, seed in enumerate(seeds):
        print(f"  seed {seed} ({i + 1}/{n_seeds}) ...", end=" ", flush=True)
        res = run_one_seed(df, config, scheme, seed, epsilons, delta, mechanism)
        for model_name in MODELS:
            all_runs[model_name]["no_privacy"].append(res[model_name]["no_privacy"])
            for eps in epsilons:
                all_runs[model_name]["dp"][str(eps)].append(res[model_name]["dp"][str(eps)])
        print("done")

    # --- aggregate to mean ± std ---
    print("\n" + "=" * 74)
    print(f"DP PRIVACY-UTILITY SWEEP -- {n_seeds} SEEDS (mean ± std)")
    print("=" * 74)

    def agg(runs: list, key: str) -> tuple[float, float]:
        vals = np.array([r[key] for r in runs])
        return float(vals.mean()), float(vals.std())

    results: dict = {}
    for model_name in MODELS:
        print(f"\nMODEL: {model_name}")
        no_priv_runs = all_runs[model_name]["no_privacy"]
        acc_m, acc_s = agg(no_priv_runs, "accuracy")
        rec_m, rec_s = agg(no_priv_runs, "recall")
        auc_m, auc_s = agg(no_priv_runs, "auc_roc")
        print(f"  no privacy: accuracy {acc_m:.4f} ± {acc_s:.4f}")

        results[model_name] = {
            "no_privacy": {
                "accuracy_mean": acc_m, "accuracy_std": acc_s,
                "recall_mean": rec_m, "recall_std": rec_s,
                "auc_mean": auc_m, "auc_std": auc_s,
            },
            "dp": {},
        }

        for eps in epsilons:
            runs = all_runs[model_name]["dp"][str(eps)]
            acc_m, acc_s = agg(runs, "accuracy")
            rec_m, rec_s = agg(runs, "recall")
            auc_m, auc_s = agg(runs, "auc_roc")
            cost_m, cost_s = agg(runs, "privacy_cost")
            print(f"  epsilon={eps}: accuracy {acc_m:.4f} ± {acc_s:.4f}   "
                  f"privacy cost {cost_m:+.4f} ± {cost_s:.4f}")
            results[model_name]["dp"][str(eps)] = {
                "accuracy_mean": acc_m, "accuracy_std": acc_s,
                "recall_mean": rec_m, "recall_std": rec_s,
                "auc_mean": auc_m, "auc_std": auc_s,
                "privacy_cost_mean": cost_m, "privacy_cost_std": cost_s,
            }

    # --- privacy-utility trade-off plot ---
    figures_dir = ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    eps_vals = sorted(float(e) for e in epsilons)
    for model_name in MODELS:
        accs = np.array([results[model_name]["dp"][str(e)]["accuracy_mean"] for e in eps_vals])
        stds = np.array([results[model_name]["dp"][str(e)]["accuracy_std"] for e in eps_vals])
        plt.plot(eps_vals, accs, marker="o", label=model_name)
        plt.fill_between(eps_vals, accs - stds, accs + stds, alpha=0.15)
        plt.axhline(results[model_name]["no_privacy"]["accuracy_mean"],
                    linestyle="--", alpha=0.4)

    plt.xlabel("Privacy budget ε  (smaller = stronger privacy)")
    plt.ylabel("Accuracy")
    plt.title(f"Privacy–Utility Trade-off (Differential Privacy, {n_seeds} seeds, mean ± std)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = figures_dir / "privacy_utility_dp.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\n✓ Saved privacy–utility plot: {out_fig}")

    out = ROOT / config["paths"]["metrics_dir"] / "privacy_dp.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"✓ Saved DP results: {out}")


if __name__ == "__main__":
    main()
