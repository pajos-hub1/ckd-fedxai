"""Stage 8c: homomorphic encryption on the Logistic Regression track (§3.7.2).

Run from the project root:
    python scripts/07_privacy_he.py

Runs the SAME FedAvg-LR training as Stage 8b (05c_federated_lr.py), on the
same fixed on-disk partitions and the same base seed, but with every
round's aggregation performed under CKKS homomorphic encryption instead
of plaintext averaging. Because local training is identical, this run
isolates exactly what HE costs: NOT accuracy (there is no noise, unlike
DP) but the computational and communication overhead of ciphertext
arithmetic -- encryption time, homomorphic-aggregation time, decryption
time, and ciphertext size vs the plaintext weight vector (§3.7.3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.evaluation.metrics import compute_metrics
from ckd_fedxai.federated.fedavg_lr import predict_proba_from_weights, train_fedavg_lr
from ckd_fedxai.federated.simulation import load_clients
from ckd_fedxai.privacy.homomorphic import train_fedavg_lr_he
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed


def evaluate_history(history: list, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    X_arr = X_test.to_numpy()
    per_round = []
    for coef, intercept in history:
        proba = predict_proba_from_weights(coef, intercept, X_arr)
        pred = (proba >= 0.5).astype(int)
        per_round.append(compute_metrics(y_test, pred, proba))
    return per_round


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    target = config["data"]["target_column"]
    partitions_dir = ROOT / config["paths"]["partitions_dir"]
    scheme = config["federated"]["partition_scheme"]
    base_seed = config["seed"]
    num_rounds = config["federated"]["num_rounds"]

    test_df = pd.read_csv(partitions_dir / "test.csv")
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    X_clients, y_clients, sizes = load_clients(partitions_dir, scheme, target)

    he_cfg = config["privacy"]["homomorphic_encryption"]
    print(f"Partition scheme: {scheme}, base seed: {base_seed}, rounds: {num_rounds}")
    print(f"CKKS: poly_modulus_degree={he_cfg['poly_modulus_degree']}, "
          f"coeff_mod_bit_sizes={he_cfg['coeff_mod_bit_sizes']}\n")

    # --- plaintext FedAvg-LR, timed for a direct overhead comparison ---
    print("Running plaintext FedAvg-LR (timed reference)...")
    t0 = time.perf_counter()
    _, _, plain_history = train_fedavg_lr(X_clients, y_clients, sizes, config, base_seed)
    plain_wall_seconds = time.perf_counter() - t0
    plain_metrics = evaluate_history(plain_history, X_test, y_test)
    print(f"  done in {plain_wall_seconds:.3f}s, "
          f"final accuracy={plain_metrics[-1]['accuracy']:.4f}\n")

    # --- HE-aggregated FedAvg-LR ---
    print("Running HE-aggregated FedAvg-LR...")
    t0 = time.perf_counter()
    he_coef, he_intercept, he_history, overheads = train_fedavg_lr_he(
        X_clients, y_clients, sizes, config, base_seed
    )
    he_wall_seconds = time.perf_counter() - t0
    he_metrics = evaluate_history(he_history, X_test, y_test)
    print(f"  done in {he_wall_seconds:.3f}s, "
          f"final accuracy={he_metrics[-1]['accuracy']:.4f}\n")

    # --- accuracy comparison: HE should add ~no loss vs plaintext ---
    acc_diff = plain_metrics[-1]["accuracy"] - he_metrics[-1]["accuracy"]
    print("=" * 66)
    print("ACCURACY: plaintext vs HE (should be ~0 -- HE adds no noise)")
    print("=" * 66)
    print(f"  plaintext final accuracy: {plain_metrics[-1]['accuracy']:.6f}")
    print(f"  HE final accuracy:        {he_metrics[-1]['accuracy']:.6f}")
    print(f"  difference:               {acc_diff:+.6f}  "
          f"(CKKS numerical precision, not a privacy-utility trade-off)")

    # --- overhead summary ---
    enc_times = np.array([o.encrypt_seconds for o in overheads])
    agg_times = np.array([o.aggregate_seconds for o in overheads])
    dec_times = np.array([o.decrypt_seconds for o in overheads])
    ct_bytes = overheads[0].ciphertext_bytes
    pt_bytes = overheads[0].plaintext_bytes

    print("\n" + "=" * 66)
    print("COMPUTATIONAL / COMMUNICATION OVERHEAD (§3.7.3)")
    print("=" * 66)
    print(f"  per-round encrypt time (all clients): {enc_times.mean()*1000:.2f} ms "
          f"± {enc_times.std()*1000:.2f} ms")
    print(f"  per-round aggregate time (server):    {agg_times.mean()*1000:.2f} ms "
          f"± {agg_times.std()*1000:.2f} ms")
    print(f"  per-round decrypt time:               {dec_times.mean()*1000:.2f} ms "
          f"± {dec_times.std()*1000:.2f} ms")
    print(f"  total wall time -- plaintext FedAvg:   {plain_wall_seconds:.3f} s")
    print(f"  total wall time -- HE FedAvg:          {he_wall_seconds:.3f} s")
    print(f"  slowdown factor:                       {he_wall_seconds / plain_wall_seconds:.1f}x")
    print(f"  ciphertext size (one client's update): {ct_bytes:,} bytes")
    print(f"  plaintext size (one client's update):  {pt_bytes:,} bytes")
    print(f"  size blow-up factor:                   {ct_bytes / pt_bytes:.1f}x")

    # --- convergence plot: plaintext vs HE overlay ---
    figures_dir = ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    round_ids = np.arange(1, num_rounds + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(round_ids, [m["accuracy"] for m in plain_metrics],
              marker="o", label="plaintext FedAvg-LR")
    plt.plot(round_ids, [m["accuracy"] for m in he_metrics],
              marker="x", linestyle="--", label="HE-aggregated FedAvg-LR")
    plt.xlabel("FedAvg round")
    plt.ylabel("Accuracy (held-out test set)")
    plt.title(f"Plaintext vs HE-Aggregated FedAvg-LR ({scheme} partition, seed={base_seed})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_fig = figures_dir / "he_vs_plaintext_convergence.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\n✓ Saved convergence comparison plot: {out_fig}")

    # --- overhead bar chart: ciphertext vs plaintext size ---
    plt.figure(figsize=(6, 5))
    plt.bar(["plaintext", "ciphertext"], [pt_bytes, ct_bytes],
            color=["tab:blue", "tab:red"])
    plt.ylabel("Bytes (one client's weight update)")
    plt.title("HE Communication Overhead:\nCiphertext vs Plaintext Size")
    plt.yscale("log")
    plt.tight_layout()
    out_fig2 = figures_dir / "he_size_overhead.png"
    plt.savefig(out_fig2, dpi=150)
    plt.close()
    print(f"✓ Saved size-overhead plot: {out_fig2}")

    # --- save results JSON ---
    results = {
        "scheme": scheme,
        "seed": base_seed,
        "num_rounds": num_rounds,
        "ckks_params": he_cfg,
        "plaintext_final_accuracy": plain_metrics[-1]["accuracy"],
        "he_final_accuracy": he_metrics[-1]["accuracy"],
        "accuracy_difference": float(acc_diff),
        "plaintext_round_accuracy": [m["accuracy"] for m in plain_metrics],
        "he_round_accuracy": [m["accuracy"] for m in he_metrics],
        "overhead": {
            "encrypt_ms_mean": float(enc_times.mean() * 1000),
            "encrypt_ms_std": float(enc_times.std() * 1000),
            "aggregate_ms_mean": float(agg_times.mean() * 1000),
            "aggregate_ms_std": float(agg_times.std() * 1000),
            "decrypt_ms_mean": float(dec_times.mean() * 1000),
            "decrypt_ms_std": float(dec_times.std() * 1000),
            "plain_wall_seconds": plain_wall_seconds,
            "he_wall_seconds": he_wall_seconds,
            "slowdown_factor": he_wall_seconds / plain_wall_seconds,
            "ciphertext_bytes_per_client": ct_bytes,
            "plaintext_bytes_per_client": pt_bytes,
            "size_blowup_factor": ct_bytes / pt_bytes,
        },
    }
    out = ROOT / config["paths"]["metrics_dir"] / "privacy_he.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"✓ Saved HE results: {out}")


if __name__ == "__main__":
    main()
