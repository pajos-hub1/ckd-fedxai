"""Stage 9: consolidated evaluation (§3.8, §4 tables/figures).

Run from the project root:
    python scripts/08_evaluate.py

Pulls together every result produced by Stages 2-8c into the tables and
figures Chapter 4 reports against, and states directly the Objective 5
payoff: DP (Track A, trees) trades ACCURACY for privacy; HE (Track B, LR)
trades COMPUTATION/COMMUNICATION for privacy, at essentially no accuracy
cost. Nothing is recomputed here -- this stage only reads the JSON already
saved by 02_train_central.py, 05b_federated_multiseed.py,
05c_federated_lr.py, 06_privacy_dp.py, and 07_privacy_he.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.utils.config import load_config


def load_metrics(config: dict, name: str) -> dict:
    path = ROOT / config["paths"]["metrics_dir"] / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run the corresponding stage script first."
        )
    return json.loads(path.read_text())


def table_predictive_performance(central: dict, fed_multiseed: dict, fed_lr: dict) -> str:
    """Table 4.1: centralised vs federated, across models and partition schemes."""
    lines = [
        "TABLE 4.1 -- PREDICTIVE PERFORMANCE: CENTRALISED vs FEDERATED",
        "=" * 90,
        f"{'Model':<20}{'Setting':<18}{'Accuracy':<20}{'Recall':<20}{'AUC-ROC':<15}",
        "-" * 90,
    ]

    for model_name in ["random_forest", "xgboost"]:
        central_test = central[model_name]["test_metrics"]
        lines.append(
            f"{model_name:<20}{'centralised':<18}"
            f"{central_test['accuracy']:<20.4f}{central_test['recall']:<20.4f}"
            f"{central_test['auc_roc']:<15.4f}"
        )
        ms = fed_multiseed[model_name]
        for scheme in ["iid", "non_iid_equal", "non_iid"]:
            s = ms[scheme]
            lines.append(
                f"{'':<20}{scheme:<18}"
                f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}   "
                f"{s['recall_mean']:.4f} ± {s['recall_std']:.4f}   "
                f"{s['auc_mean']:.4f}"
            )
        lines.append("-" * 90)

    # LR / FedAvg track
    central_lr = central["logistic_regression"]["test_metrics"]
    lines.append(
        f"{'logistic_reg':<20}{'centralised':<18}"
        f"{central_lr['accuracy']:<20.4f}{central_lr['recall']:<20.4f}"
        f"{central_lr['auc_roc']:<15.4f}"
    )
    ms_lr = fed_lr["multiseed_final_round"]
    lines.append(
        f"{'':<20}{'fedavg (' + fed_lr['scheme'] + ')':<18}"
        f"{ms_lr['accuracy_mean']:.4f} ± {ms_lr['accuracy_std']:.4f}   "
        f"{ms_lr['recall_mean']:.4f} ± {ms_lr['recall_std']:.4f}   "
        f"{ms_lr['auc_mean']:.4f}"
    )
    lines.append("-" * 90)

    # federation cost summary
    lines.append("\nFEDERATION COST (centralised accuracy - federated accuracy):")
    for model_name in ["random_forest", "xgboost"]:
        central_acc = central[model_name]["test_metrics"]["accuracy"]
        for scheme in ["iid", "non_iid_equal", "non_iid"]:
            fed_acc = fed_multiseed[model_name][scheme]["accuracy_mean"]
            lines.append(f"  {model_name:<18}{scheme:<16}{central_acc - fed_acc:+.4f}")
    lines.append(
        f"  {'logistic_reg':<18}{'fedavg':<16}"
        f"{central_lr['accuracy'] - ms_lr['accuracy_mean']:+.4f}"
    )
    return "\n".join(lines)


def table_dp_privacy_utility(dp: dict) -> str:
    """Table 4.2: DP privacy-utility trade-off (Track A)."""
    lines = [
        "\nTABLE 4.2 -- DIFFERENTIAL PRIVACY: PRIVACY-UTILITY TRADE-OFF (TRACK A)",
        "=" * 78,
        f"{'Model':<18}{'Epsilon':<12}{'Accuracy':<14}{'Privacy Cost':<16}",
        "-" * 78,
    ]
    for model_name in ["random_forest", "xgboost"]:
        base = dp[model_name]["no_privacy"]["accuracy"]
        lines.append(f"{model_name:<18}{'inf (none)':<12}{base:<14.4f}{'—':<16}")
        eps_sorted = sorted(dp[model_name]["dp"].keys(), key=float, reverse=True)
        for eps in eps_sorted:
            row = dp[model_name]["dp"][eps]
            lines.append(
                f"{'':<18}{eps:<12}{row['accuracy']:<14.4f}{row['privacy_cost']:<+16.4f}"
            )
        lines.append("-" * 78)
    return "\n".join(lines)


def table_he_overhead(he: dict) -> str:
    """Table 4.3: HE overhead (Track B) -- no accuracy cost, all cost is compute/comms."""
    o = he["overhead"]
    lines = [
        "\nTABLE 4.3 -- HOMOMORPHIC ENCRYPTION: COMPUTATIONAL/COMMUNICATION OVERHEAD (TRACK B)",
        "=" * 78,
        f"  plaintext FedAvg-LR accuracy:  {he['plaintext_final_accuracy']:.4f}",
        f"  HE-aggregated FedAvg-LR accuracy: {he['he_final_accuracy']:.4f}",
        f"  accuracy difference:           {he['accuracy_difference']:+.6f}  (numerical only)",
        "-" * 78,
        f"  encrypt time / round (all clients): {o['encrypt_ms_mean']:.2f} ms ± {o['encrypt_ms_std']:.2f} ms",
        f"  aggregate time / round (server):    {o['aggregate_ms_mean']:.2f} ms ± {o['aggregate_ms_std']:.2f} ms",
        f"  decrypt time / round:                {o['decrypt_ms_mean']:.2f} ms ± {o['decrypt_ms_std']:.2f} ms",
        f"  total wall time, plaintext:          {o['plain_wall_seconds']:.3f} s",
        f"  total wall time, HE:                 {o['he_wall_seconds']:.3f} s",
        f"  slowdown factor:                     {o['slowdown_factor']:.1f}x",
        f"  ciphertext size / client update:     {o['ciphertext_bytes_per_client']:,} bytes",
        f"  plaintext size / client update:      {o['plaintext_bytes_per_client']:,} bytes",
        f"  size blow-up factor:                 {o['size_blowup_factor']:.1f}x",
        "=" * 78,
    ]
    return "\n".join(lines)


def plot_objective5(dp: dict, he: dict, out_path: Path) -> None:
    """The Objective 5 payoff figure: DP costs accuracy (Track A) vs HE
    costs computation/communication (Track B), side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel A: DP privacy-utility curve ---
    ax = axes[0]
    for model_name in ["random_forest", "xgboost"]:
        eps_vals = sorted(float(e) for e in dp[model_name]["dp"].keys())
        accs = [dp[model_name]["dp"][str(e)]["accuracy"] for e in eps_vals]
        ax.plot(eps_vals, accs, marker="o", label=model_name)
        ax.axhline(dp[model_name]["no_privacy"]["accuracy"], linestyle="--", alpha=0.3)
    ax.set_xlabel("Privacy budget ε (smaller = stronger privacy)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Track A (DP): privacy costs ACCURACY")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel B: HE overhead ---
    ax = axes[1]
    o = he["overhead"]
    labels = ["wall time\n(slowdown ×)", "message size\n(blow-up ×)"]
    values = [o["slowdown_factor"], o["size_blowup_factor"]]
    bars = ax.bar(labels, values, color=["tab:orange", "tab:red"])
    ax.set_yscale("log")
    ax.set_ylabel("Factor vs plaintext (log scale)")
    ax.set_title(
        f"Track B (HE): privacy costs COMPUTATION\n"
        f"(accuracy difference: {he['accuracy_difference']:+.6f})"
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}×",
                 ha="center", va="bottom")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Objective 5: Privacy-Utility-Efficiency Trade-off "
                 "(DP on ensembles vs HE on logistic regression)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = load_config()

    central = load_metrics(config, "central_baseline.json")
    fed_multiseed = load_metrics(config, "federated_multiseed.json")
    fed_lr = load_metrics(config, "federated_lr.json")
    dp = load_metrics(config, "privacy_dp.json")
    he = load_metrics(config, "privacy_he.json")

    t1 = table_predictive_performance(central, fed_multiseed, fed_lr)
    t2 = table_dp_privacy_utility(dp)
    t3 = table_he_overhead(he)

    print(t1)
    print(t2)
    print(t3)

    figures_dir = ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "objective5_privacy_utility_efficiency.png"
    plot_objective5(dp, he, fig_path)
    print(f"\n✓ Saved Objective 5 figure: {fig_path}")

    # --- markdown export of the three tables, ready to paste into Ch4 ---
    md_path = ROOT / config["paths"]["metrics_dir"] / "ch4_tables.md"
    md_path.write_text(
        "# Chapter 4 -- Consolidated Results\n\n"
        "```\n" + t1 + "\n```\n\n"
        "```\n" + t2 + "\n```\n\n"
        "```\n" + t3 + "\n```\n"
    )
    print(f"✓ Saved Ch4 tables (markdown): {md_path}")

    # --- single consolidated JSON for programmatic reuse ---
    consolidated = {
        "central_baseline": central,
        "federated_multiseed": fed_multiseed,
        "federated_lr": fed_lr,
        "privacy_dp": dp,
        "privacy_he": he,
    }
    out = ROOT / config["paths"]["metrics_dir"] / "consolidated_evaluation.json"
    out.write_text(json.dumps(consolidated, indent=2))
    print(f"✓ Saved consolidated evaluation: {out}")


if __name__ == "__main__":
    main()
