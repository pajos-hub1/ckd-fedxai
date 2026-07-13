"""Data partitioning into simulated hospital nodes (§3.4.2).

Two schemes:
  - IID:     data shuffled and split uniformly; each client's distribution
             resembles the whole (idealised case, similar hospitals).
  - non-IID: Dirichlet-controlled label skew; clients receive different
             proportions of CKD/non-CKD cases (realistic heterogeneity).

The train/test split is done FIRST, so the held-out test set stays global
and identical to the centralised baseline — this keeps federated and
centralised results directly comparable (§3.8.3, federation cost).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def iid_partition(df: pd.DataFrame, num_clients: int, seed: int) -> list[pd.DataFrame]:
    """Shuffle and split uniformly at random across clients."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # split by index positions so we get DataFrames back, not ndarrays
    index_splits = np.array_split(np.arange(len(shuffled)), num_clients)
    return [shuffled.iloc[idx].reset_index(drop=True) for idx in index_splits]


def non_iid_partition(df: pd.DataFrame, num_clients: int, target: str,
                      alpha: float, seed: int) -> list[pd.DataFrame]:
    """Dirichlet label-skew partition.

    For each class, the samples are divided among clients in proportions
    drawn from a Dirichlet(alpha) distribution. Lower alpha => more skew
    (some clients get mostly CKD, others mostly non-CKD).
    """
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in sorted(df[target].unique()):
        # .copy() -> writable array (to_numpy() can return a read-only view)
        cls_idx = df.index[df[target] == cls].to_numpy().copy()
        rng.shuffle(cls_idx)

        # proportions of this class going to each client
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, cuts)

        for c, split in enumerate(splits):
            client_indices[c].extend(split.tolist())

    parts = []
    for idx in client_indices:
        idx_arr = np.array(idx)
        rng.shuffle(idx_arr)
        parts.append(df.loc[idx_arr].reset_index(drop=True))
    return parts


def partition(df: pd.DataFrame, config: dict, scheme: str | None = None
              ) -> list[pd.DataFrame]:
    """Partition according to the config scheme (or an override)."""
    fed = config["federated"]
    scheme = scheme or fed["partition_scheme"]
    num_clients = fed["num_clients"]
    seed = config["seed"]
    target = config["data"]["target_column"]

    if scheme == "iid":
        return iid_partition(df, num_clients, seed)
    elif scheme == "non_iid":
        return non_iid_partition(df, num_clients, target,
                                 fed["dirichlet_alpha"], seed)
    raise ValueError(f"Unknown partition scheme: {scheme!r}")


def summarise(parts: list[pd.DataFrame], target: str, scheme: str) -> None:
    """Print per-client size and class balance — shows the heterogeneity."""
    print(f"\n{scheme.upper()} PARTITION — {len(parts)} simulated hospitals")
    print(f"{'Client':<10}{'N':<8}{'CKD':<8}{'non-CKD':<10}{'% CKD':<8}")
    print("-" * 44)
    for i, part in enumerate(parts):
        n = len(part)
        ckd = int((part[target] == 1).sum())
        notckd = int((part[target] == 0).sum())
        pct = 100 * ckd / n if n else 0
        print(f"{i+1:<10}{n:<8}{ckd:<8}{notckd:<10}{pct:<8.1f}")
    print("-" * 44)
    total = sum(len(p) for p in parts)
    print(f"{'TOTAL':<10}{total:<8}")