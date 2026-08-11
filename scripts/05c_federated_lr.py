"""Stage 8b: true iterative FedAvg for Logistic Regression (§3.5.2, §3.7.2).

Run from the project root:
    python scripts/05c_federated_lr.py

Unlike Track A (trees, aggregated by output averaging), the LR track is
trained by genuine round-by-round FedAvg: each client trains locally from
the current global weights, returns its updated weight vector, and the
server size-weighted-averages the vectors before the next round. Tracks
accuracy at every round to show convergence, and repeats across multiple
seeds so the curve isn't a single lucky partition draw.

The final global weight vector (base seed, configured partition scheme)
is saved to results/models/ -- this is what Stage 8c (homomorphic
encryption) will encrypt.
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

from sklearn.model_selection import train_test_split

from ckd_fedxai.data.partition import partition
from ckd_fedxai.evaluation.metrics import compute_metrics
from ckd_fedxai.federated.fedavg_lr import predict_proba_from_weights, train_fedavg_lr
from ckd_fedxai.federated.simulation import load_clients
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed


def evaluate_history(history: list, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    """Accuracy/AUC at every FedAvg round, for the convergence curve."""
    X_arr = X_test.to_numpy()
    per_round = []
    for coef, intercept in history:
        proba = predict_proba_from_weights(coef, intercept, X_arr)
        pred = (proba >= 0.5).astype(int)
        per_round.append(compute_metrics(y_test, pred, proba))
    return per_round


def run_canonical(config: dict, scheme: str, seed: int):
    """The single reference run, on the FIXED on-disk partitions and the
    FIXED global test.csv -- the same convention 06_privacy_dp.py uses, so
    Stage 9 can compare DP (Track A) and HE (Track B) on identical data.
    Produces the convergence plot and the weight vector Stage 8c encrypts.
    """
    target = config["data"]["target_column"]
    partitions_dir = ROOT / config["paths"]["partitions_dir"]

    test_df = pd.read_csv(partitions_dir / "test.csv")
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    X_clients, y_clients, sizes = load_clients(partitions_dir, scheme, target)
    coef, intercept, history = train_fedavg_lr(X_clients, y_clients, sizes, config, seed)
    per_round = evaluate_history(history, X_test, y_test)
    return coef, intercept, per_round


def run_one_seed(df: pd.DataFrame, config: dict, scheme: str, seed: int):
    """One full experiment at a given seed: fresh stratified split, fresh
    partition, FedAvg training, evaluation on that seed's own test split.

    Mirrors 05b_federated_multiseed.py's convention -- re-splitting and
    re-partitioning per seed is what actually produces genuine seed-to-
    seed variability (re-using the fixed on-disk partitions would only
    vary SGD's internal shuffle order, which barely moves the decision
    boundary on this small, separable dataset).
    """
    target = config["data"]["target_column"]
    train_df, test_df = train_test_split(
        df, test_size=config["data"]["test_size"],
        stratify=df[target], random_state=seed,
    )
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    parts = partition(train_df, config, scheme=scheme)
    X_clients = [p.drop(columns=[target]) for p in parts]
    y_clients = [p[target] for p in parts]
    sizes = [len(p) for p in parts]

    coef, intercept, history = train_fedavg_lr(X_clients, y_clients, sizes, config, seed)
    per_round = evaluate_history(history, X_test, y_test)
    return per_round[-1]


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    target = config["data"]["target_column"]

    scheme = config["federated"]["partition_scheme"]
    num_rounds = config["federated"]["num_rounds"]
    n_seeds = config["federated"].get("num_seeds", 10)
    base_seed = config["seed"]
    seeds = [base_seed + i for i in range(n_seeds)]

    print(f"Partition scheme: {scheme}")
    print(f"Rounds per run: {num_rounds}, "
          f"local epochs: {config['federated']['local_epochs']}")
    print(f"Seeds: {seeds}\n")

    # --- canonical run: fixed on-disk partitions + fixed global test.csv,
    #     same convention as 06_privacy_dp.py, so Stage 9 can compare DP
    #     (Track A) and HE (Track B) results on identical data ---
    print("Canonical run (fixed partitions, base seed) ...")
    base_seed_coef, base_seed_intercept, canonical_history = run_canonical(
        config, scheme, base_seed
    )
    print(f"  final accuracy: {canonical_history[-1]['accuracy']:.4f}\n")

    # --- multi-seed robustness: fresh split + repartition per seed,
    #     mirroring 05b_federated_multiseed.py's convention ---
    print(f"Multi-seed robustness check ({n_seeds} seeds, fresh split+partition each)...")
    df = pd.read_csv(ROOT / config["paths"]["processed_data"])

    final_metrics_per_seed = []
    for i, seed in enumerate(seeds):
        print(f"  seed {seed} ({i + 1}/{n_seeds}) ...", end=" ", flush=True)
        final = run_one_seed(df, config, scheme, seed)
        final_metrics_per_seed.append(final)
        print(f"final acc={final['accuracy']:.4f}")

    # --- final-round summary across seeds ---
    print("\n" + "=" * 66)
    print(f"FEDAVG-LR FINAL ROUND -- {n_seeds} SEEDS (mean ± std)")
    print("=" * 66)
    final_acc = np.array([m["accuracy"] for m in final_metrics_per_seed])
    final_rec = np.array([m["recall"] for m in final_metrics_per_seed])
    final_auc = np.array([m["auc_roc"] for m in final_metrics_per_seed])
    print(f"  accuracy: {final_acc.mean():.4f} ± {final_acc.std():.4f}")
    print(f"  recall:   {final_rec.mean():.4f} ± {final_rec.std():.4f}")
    print(f"  auc_roc:  {final_auc.mean():.4f} ± {final_auc.std():.4f}")

    # --- compare against the centralised LR baseline, if available ---
    central_path = ROOT / config["paths"]["metrics_dir"] / "central_baseline.json"
    central_lr_acc = None
    if central_path.exists():
        central = json.loads(central_path.read_text())
        if "logistic_regression" in central:
            central_lr_acc = central["logistic_regression"]["test_metrics"]["accuracy"]
            print(f"\n  centralised LR accuracy: {central_lr_acc:.4f}")
            print(f"  federation cost:         {central_lr_acc - final_acc.mean():+.4f}")

    # --- convergence plot: canonical (fixed-partition) run, round by round ---
    figures_dir = ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    round_ids = np.arange(1, num_rounds + 1)
    canonical_acc = np.array([m["accuracy"] for m in canonical_history])

    plt.figure(figsize=(7, 5))
    plt.plot(round_ids, canonical_acc, marker="o", label="FedAvg-LR accuracy")
    plt.axhline(final_acc.mean(), linestyle=":", color="tab:orange", alpha=0.7,
                label=f"multi-seed final mean ({n_seeds} seeds)")
    if central_lr_acc is not None:
        plt.axhline(central_lr_acc, linestyle="--", color="grey", alpha=0.6,
                    label="centralised LR")
    plt.xlabel("FedAvg round")
    plt.ylabel("Accuracy (held-out test set)")
    plt.title(f"FedAvg-LR Convergence ({scheme} partition, seed={base_seed})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = figures_dir / "fedavg_lr_convergence.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\n✓ Saved convergence plot: {out_fig}")

    # --- save results JSON ---
    results = {
        "scheme": scheme,
        "num_rounds": num_rounds,
        "seeds": seeds,
        "canonical_round_accuracy": canonical_acc.tolist(),
        "canonical_round_auc": [m["auc_roc"] for m in canonical_history],
        "multiseed_final_round": {
            "accuracy_mean": float(final_acc.mean()),
            "accuracy_std": float(final_acc.std()),
            "recall_mean": float(final_rec.mean()),
            "recall_std": float(final_rec.std()),
            "auc_mean": float(final_auc.mean()),
            "auc_std": float(final_auc.std()),
        },
        "centralised_lr_accuracy": central_lr_acc,
    }
    out = ROOT / config["paths"]["metrics_dir"] / "federated_lr.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"✓ Saved FedAvg-LR results: {out}")

    # --- save the canonical global weight vector for Stage 8c (HE) ---
    models_dir = ROOT / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    feature_names = df.drop(columns=[target]).columns.tolist()
    out_weights = models_dir / "fedavg_lr_weights.npz"
    np.savez(out_weights, coef=base_seed_coef, intercept=base_seed_intercept,
             feature_names=np.array(feature_names))
    print(f"✓ Saved global LR weight vector for HE (Stage 8c): {out_weights}")


if __name__ == "__main__":
    main()
