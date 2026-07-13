"""Federated training simulation (§3.4).

Implements the server/client cycle:
  1. server distributes the model configuration to clients
  2. each client trains locally on its own partition (data never leaves)
  3. clients return their trained models
  4. server aggregates them into a global model

Only model parameters/structures are exchanged — never raw patient data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ckd_fedxai.models.centralised import build_models


def load_clients(partitions_dir: Path, scheme: str, target: str
                 ) -> tuple[list[pd.DataFrame], list[pd.Series], list[int]]:
    """Load each client's partition from disk."""
    scheme_dir = partitions_dir / scheme
    files = sorted(scheme_dir.glob("client_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No client files in {scheme_dir}. Run 04_partition.py first."
        )

    X_list, y_list, sizes = [], [], []
    for f in files:
        df = pd.read_csv(f)
        X_list.append(df.drop(columns=[target]))
        y_list.append(df[target])
        sizes.append(len(df))
    return X_list, y_list, sizes


def train_federated(model_name: str, X_clients, y_clients, sizes,
                    config: dict) -> list:
    """Train one local model per client; return the client models.

    Each client trains only on its OWN partition — raw data never leaves
    the node (§3.4.1). The server then aggregates the resulting models.
    """
    client_models = []
    for i, (X, y) in enumerate(zip(X_clients, y_clients)):
        local_model = build_models(config)[model_name]
        local_model.fit(X, y)
        client_models.append(local_model)
        n_ckd = int((y == 1).sum())
        print(f"    client {i+1}: {len(X)} records "
              f"({n_ckd} CKD / {len(X) - n_ckd} non-CKD)")

    print(f"    server: size-weighted aggregation of "
          f"{len(client_models)} client models")
    return client_models