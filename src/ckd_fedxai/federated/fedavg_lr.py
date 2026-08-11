"""True iterative FedAvg for the Logistic Regression track (§3.5.2, §3.7.2).

Unlike the tree ensembles of Track A, logistic regression IS a numeric
weight vector (coefficients + intercept), so it can be aggregated the way
Equation 3.1 originally specifies: parameter-level averaging, round by
round, not the output-level averaging used for trees. This is precisely
why LR is the track chosen to carry homomorphic encryption in Stage 8c --
HE encrypts numeric ciphertexts, and a weight vector is the only model
representation here that can be encrypted, aggregated while encrypted,
and decrypted back into a usable model.

Each federated round:
  1. server broadcasts the current global (coef, intercept)
  2. every client initialises a local model from those global weights and
     trains `local_epochs` further epochs of SGD on its own partition
     (raw data never leaves the client)
  3. clients return their updated (coef, intercept)
  4. server aggregates by size-weighted averaging (n_k / n) -- the same
     weighting used by the tree track's output aggregation
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier


def build_local_lr(config: dict, seed: int) -> SGDClassifier:
    """A linear model trained by SGD.

    SGDClassifier is used instead of LogisticRegression's default lbfgs
    solver because its coef_/intercept_ can be warm-started from
    externally supplied values and advanced by a fixed number of epochs --
    exactly what one FedAvg round requires. lbfgs re-optimises to
    convergence regardless of the starting point, which would erase the
    global weights broadcast at the start of the round.
    """
    lr_cfg = config["models"]["logistic_regression_fedavg"]
    epochs = config["federated"]["local_epochs"]
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=lr_cfg["alpha"],
        learning_rate="constant",
        eta0=lr_cfg["learning_rate"],
        max_iter=epochs,
        tol=None,           # disable early stopping: always run exactly
        warm_start=True,    # `epochs` more passes from the injected weights
        random_state=seed,
    )


def init_global_weights(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-initialised global weight vector (round 0)."""
    return np.zeros(n_features), np.zeros(1)


def set_weights(model: SGDClassifier, coef: np.ndarray, intercept: np.ndarray,
                classes: np.ndarray) -> None:
    """Inject the current global weights into a fresh local model so its
    local epochs of SGD continue from the global state, not from scratch.
    """
    model.classes_ = classes
    model.coef_ = coef.reshape(1, -1).copy()
    model.intercept_ = intercept.copy()


def federated_round(global_coef: np.ndarray, global_intercept: np.ndarray,
                    X_clients: list, y_clients: list, sizes: list[int],
                    config: dict, seed: int, classes: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One FedAvg round: local training on every client, then size-weighted
    averaging of the resulting weight vectors (Equation 3.1, applied at
    the parameter level rather than the output level).
    """
    total = sum(sizes)
    client_coefs, client_intercepts = [], []

    for X, y in zip(X_clients, y_clients):
        local = build_local_lr(config, seed)
        set_weights(local, global_coef, global_intercept, classes)
        local.fit(X, y)
        client_coefs.append(local.coef_.ravel())
        client_intercepts.append(local.intercept_.copy())

    new_coef = sum(c * (n / total) for c, n in zip(client_coefs, sizes))
    new_intercept = sum(b * (n / total) for b, n in zip(client_intercepts, sizes))
    return new_coef, new_intercept


def train_fedavg_lr(X_clients: list, y_clients: list, sizes: list[int],
                    config: dict, seed: int
                    ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run the full FedAvg loop for `federated.num_rounds` rounds.

    Returns the converged global (coef, intercept) together with the
    per-round history of weights, so the caller can evaluate accuracy at
    every round and plot the convergence curve.
    """
    n_features = X_clients[0].shape[1]
    classes = np.array([0, 1])
    coef, intercept = init_global_weights(n_features)

    num_rounds = config["federated"]["num_rounds"]
    history = []
    for r in range(num_rounds):
        coef, intercept = federated_round(
            coef, intercept, X_clients, y_clients, sizes, config, seed + r, classes
        )
        history.append((coef.copy(), intercept.copy()))

    return coef, intercept, history


def predict_proba_from_weights(coef: np.ndarray, intercept: np.ndarray, X) -> np.ndarray:
    """Sigmoid(X @ coef + intercept): the positive-class probability from a
    bare (coef, intercept) pair, without needing a fitted sklearn object.

    Used for round-by-round convergence evaluation here, and again in
    Stage 8c to evaluate the HE-decrypted global weights.
    """
    z = np.asarray(X) @ coef + intercept
    return 1.0 / (1.0 + np.exp(-z))
