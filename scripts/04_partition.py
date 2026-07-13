"""Stage 6: partition data into simulated hospital nodes (§3.4.2).

Run from the project root:
    python scripts/04_partition.py

Produces BOTH iid/ and non_iid/ partitions so the federated experiments
can compare them. The global held-out test set is saved separately and
is identical to the one used by the centralised baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.data.partition import partition, summarise
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    df = pd.read_csv(ROOT / config["paths"]["processed_data"])
    target = config["data"]["target_column"]

    # --- global train/test split (SAME seed & stratify as baseline) ---
    train_df, test_df = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        stratify=df[target],
        random_state=config["seed"],
    )
    print(f"Global train pool: {len(train_df)} rows")
    print(f"Global held-out test: {len(test_df)} rows "
          f"(shared by all clients — same as centralised baseline)")

    out_root = ROOT / config["paths"]["partitions_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    # save the global test set once
    test_path = out_root / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✓ Saved global test set: {test_path.name}")

    # --- build BOTH partition schemes ---
    for scheme in ["iid", "non_iid"]:
        parts = partition(train_df, config, scheme=scheme)
        summarise(parts, target, scheme)

        scheme_dir = out_root / scheme
        scheme_dir.mkdir(parents=True, exist_ok=True)
        for i, part in enumerate(parts):
            part.to_csv(scheme_dir / f"client_{i+1}.csv", index=False)
        print(f"✓ Saved {len(parts)} client files to: {scheme_dir}")

    print(f"\n✓ Partitioning complete. Files in: {out_root}")


if __name__ == "__main__":
    main()