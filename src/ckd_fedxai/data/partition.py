"""Data partitioning into simulated hospital nodes (§3.4.2).

Three schemes:
  - iid            : shuffled uniform split; each client resembles the whole
  - non_iid        : Dirichlet label skew AND size skew (realistic messy case)
  - non_iid_equal  : Dirichlet label skew with EQUAL client sizes
                     (isolates the distribution effect from the size effect)

The train/test split is done FIRST, so the held-out test set stays global
and identical to the centralised baseline — keeping federated and
centralised results directly comparable (§3.8.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def iid_partition(df: pd.DataFrame, num_clients: int, seed: int) -> list[pd.DataFrame]:
    """Shuffle and split uniformly at random across clients."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    index_splits = np.array_split(np.arange(len(shuffled)), num_clients)
    return [shuffled.iloc[idx].reset_index(drop=True) for idx in index_splits]


def non_iid_partition(df: pd.DataFrame, num_clients: int, target: str,
                      alpha: float, seed: int) -> list[pd.DataFrame]:
    """Dirichlet label-skew partition (sizes also vary)."""
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in sorted(df[target].unique()):
        cls_idx = df.index[df[target] == cls].to_numpy().copy()
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, cuts)

        for c, split in enumerate(splits):
            client_indices[c].extend(split.tolist())

    parts = []
    for idx in client_indices:
        arr = np.array(idx)
        rng.shuffle(arr)
        parts.append(df.loc[arr].reset_index(drop=True))
    return parts


def non_iid_equal_size(df: pd.DataFrame, num_clients: int, target: str,
                       alpha: float, seed: int, min_per_class: int = 5
                       ) -> list[pd.DataFrame]:
    """Non-IID partition with EQUAL client sizes and BOTH classes guaranteed.

    Every client receives the same number of records, but the CKD/non-CKD
    mix differs between them (Dirichlet label skew). A minimum number of
    each class is guaranteed per client so that every local model is
    trainable — a documented methodological choice (§3.4.2).
    """
    rng = np.random.default_rng(seed)
    n_per_client = len(df) // num_clients

    pools: dict = {}
    for cls in sorted(df[target].unique()):
        idx = df.index[df[target] == cls].to_numpy().copy()
        rng.shuffle(idx)
        pools[cls] = list(idx)

    classes = sorted(pools.keys())
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    # --- step 1: guarantee min_per_class of EVERY class to EVERY client ---
    for c in range(num_clients):
        for cls in classes:
            take = min(min_per_class, len(pools[cls]))
            client_indices[c].extend(pools[cls][:take])
            pools[cls] = pools[cls][take:]

    # --- step 2: fill the remainder using Dirichlet-skewed proportions ---
    mixes = rng.dirichlet([alpha] * len(classes), size=num_clients)
    for c in range(num_clients):
        remaining = n_per_client - len(client_indices[c])
        wanted = (mixes[c] * remaining).astype(int)
        for j, cls in enumerate(classes):
            take = int(min(wanted[j], len(pools[cls])))
            client_indices[c].extend(pools[cls][:take])
            pools[cls] = pools[cls][take:]

    # --- step 3: top up any short client from whatever remains ---
    leftovers = [i for cls in classes for i in pools[cls]]
    rng.shuffle(leftovers)
    for c in range(num_clients):
        while len(client_indices[c]) < n_per_client and leftovers:
            client_indices[c].append(leftovers.pop())

    parts = []
    for idx in client_indices:
        arr = np.array(idx)
        rng.shuffle(arr)
        parts.append(df.loc[arr].reset_index(drop=True))
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
    if scheme == "non_iid":
        return non_iid_partition(df, num_clients, target,
                                 fed["dirichlet_alpha"], seed)
    if scheme == "non_iid_equal":
        return non_iid_equal_size(df, num_clients, target,
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
    print(f"{'TOTAL':<10}{sum(len(p) for p in parts):<8}")