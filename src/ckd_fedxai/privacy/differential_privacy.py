"""Differential Privacy on the ensemble track (§3.7.1).

DP provides a formal guarantee that any single patient's record has a
bounded influence on the trained model. A mechanism satisfies
(epsilon, delta)-differential privacy if, for two datasets differing in a
single record, the probability of any output differs by at most a factor
governed by epsilon.

Smaller epsilon  => stronger privacy, greater distortion (lower utility)
Larger  epsilon  => weaker privacy, better preserved utility

Because tree ensembles cannot be parameter-averaged, DP is applied here by
perturbing each client's model OUTPUT (its predicted probabilities) with
calibrated noise before the server aggregates them. This protects the
information each client's update reveals about its local data, and lets
the privacy-utility trade-off be measured directly (§3.8.3).
"""
from __future__ import annotations

import numpy as np


def gaussian_noise_scale(epsilon: float, delta: float, sensitivity: float = 1.0
                         ) -> float:
    """Standard deviation of Gaussian noise for (epsilon, delta)-DP.

    Uses the classical Gaussian mechanism calibration:
        sigma >= sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon)


def laplace_noise_scale(epsilon: float, sensitivity: float = 1.0) -> float:
    """Scale parameter b of Laplace noise for epsilon-DP: b = sensitivity / epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(sensitivity / epsilon)


def privatise_predictions(proba: np.ndarray, epsilon: float, delta: float,
                          mechanism: str, rng: np.random.Generator,
                          sensitivity: float = 1.0) -> np.ndarray:
    """Add calibrated DP noise to a client's predicted probabilities.

    Probabilities lie in [0, 1], so the sensitivity of a single record's
    contribution is bounded. After adding noise the values are clipped back
    to [0, 1] so they remain valid probabilities.
    """
    n = len(proba)

    if mechanism == "gaussian":
        sigma = gaussian_noise_scale(epsilon, delta, sensitivity)
        noise = rng.normal(loc=0.0, scale=sigma, size=n)
    elif mechanism == "laplace":
        b = laplace_noise_scale(epsilon, sensitivity)
        noise = rng.laplace(loc=0.0, scale=b, size=n)
    else:
        raise ValueError(f"Unknown DP mechanism: {mechanism!r}")

    return np.clip(proba + noise, 0.0, 1.0)


def dp_weighted_aggregate(client_models: list, X, client_sizes: list[int],
                          epsilon: float, delta: float, mechanism: str,
                          seed: int, sensitivity: float = 1.0) -> np.ndarray:
    """Federated aggregation WITH differential privacy applied per client.

    Each client privatises its own predictions before they leave the node,
    so the server never sees an un-noised client output. The server then
    performs the usual size-weighted aggregation (Equation 3.1).
    """
    rng = np.random.default_rng(seed)
    total = sum(client_sizes)
    weighted = None

    for model, n in zip(client_models, client_sizes):
        proba_all = model.predict_proba(X)
        if proba_all.shape[1] == 1:
            only_class = int(model.classes_[0])
            proba = np.full(len(X), float(only_class))
        else:
            proba = proba_all[:, 1]

        # --- DP noise applied at the CLIENT, before transmission ---
        proba = privatise_predictions(proba, epsilon, delta, mechanism,
                                      rng, sensitivity)

        w = n / total
        weighted = proba * w if weighted is None else weighted + proba * w

    return weighted