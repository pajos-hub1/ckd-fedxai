"""Federated aggregation for tree-based ensembles (§3.4.3, §3.5.1).

Tree ensembles cannot be parameter-averaged in the manner of weight-based
models: a decision tree is a structure of split conditions, not a numeric
vector, so there is no element-wise average to compute. The federated
approach used here is therefore size-weighted ENSEMBLE AGGREGATION — each
client trains a local model on its own partition, and the server combines
the clients' outputs, weighting each client's contribution by the amount
of data it holds.

This is the direct analogue of the Federated Averaging weighting of
Equation 3.1, applied at the level of model outputs rather than model
parameters.

This structural property is precisely the reason that homomorphic
encryption — which operates on numeric parameter vectors — cannot be
applied to this track, motivating the two-track design set out in
§3.5.2 and §3.7.
"""
from __future__ import annotations

import numpy as np


def weighted_average_predictions(client_models: list, X, client_sizes: list[int]
                                 ) -> np.ndarray:
    """Size-weighted average of the client models' predicted probabilities.

    Each client k contributes its predicted probability weighted by
    n_k / n, where n_k is the number of records held by that client and n
    is the total across all clients — the same weighting as FedAvg
    (Equation 3.1).

    Args:
        client_models: the locally trained models, one per hospital node
        X:             the feature matrix to predict on (the held-out test set)
        client_sizes:  number of training records held by each client

    Returns:
        The aggregated probability of the positive class (CKD) for each row of X.

    Note:
        Handles the degenerate case in which a client was trained on a
        single class only. Such a model cannot discriminate and returns a
        one-column probability array; it therefore contributes its constant
        prediction. This is itself a realistic federated failure mode under
        extreme label skew.
    """
    if not client_models:
        raise ValueError("No client models to aggregate.")
    if len(client_models) != len(client_sizes):
        raise ValueError(
            f"Mismatch: {len(client_models)} models but "
            f"{len(client_sizes)} client sizes."
        )

    total = sum(client_sizes)
    if total == 0:
        raise ValueError("Total client data size is zero.")

    weighted = None

    for model, n in zip(client_models, client_sizes):
        proba_all = model.predict_proba(X)

        if proba_all.shape[1] == 1:
            # This client saw only ONE class during local training, so it
            # cannot discriminate. It contributes that constant prediction.
            only_class = int(model.classes_[0])
            proba = np.full(len(X), float(only_class))
        else:
            # standard case: probability of the positive class (CKD = 1)
            proba = proba_all[:, 1]

        w = n / total
        weighted = proba * w if weighted is None else weighted + proba * w

    return weighted