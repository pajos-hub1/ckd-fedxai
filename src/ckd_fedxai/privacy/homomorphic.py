"""Homomorphic Encryption on the Logistic Regression track (§3.7.2).

Unlike Differential Privacy, HE adds no noise to the model: it protects
each client's weight update in transit by keeping it encrypted end-to-end.
Every client's SIZE-WEIGHTED update (coef ++ intercept, scaled by n_k/n)
is encrypted under CKKS before it leaves the client; the server -- which
holds only a PUBLIC context (no secret key) -- homomorphically sums the
ciphertexts without ever seeing a plaintext weight; the encrypted sum is
then decrypted by whichever party holds the secret key, yielding the
exact same size-weighted average as the plaintext track (Equation 3.1).

Local training here is IDENTICAL to the plaintext FedAvg-LR track
(federated.fedavg_lr) -- only the aggregation step is replaced by an
encrypt / homomorphic-sum / decrypt pipeline. This isolates what HE
actually costs: not accuracy (there is no noise to trade off, as with
DP), but the COMPUTATIONAL and communication overhead of the ciphertext
arithmetic (§3.7.3) -- which is what this module measures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tenseal as ts

from ckd_fedxai.federated.fedavg_lr import build_local_lr, set_weights


def build_context(config: dict) -> ts.Context:
    """CKKS context WITH the secret key -- held by whichever party performs
    encryption and final decryption. In a real deployment this would sit
    with the clients / a trusted key-holder, never with the aggregating
    server; here it is one Python object because the whole pipeline is
    simulated on a single machine.
    """
    he_cfg = config["privacy"]["homomorphic_encryption"]
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=he_cfg["poly_modulus_degree"],
        coeff_mod_bit_sizes=he_cfg["coeff_mod_bit_sizes"],
    )
    ctx.global_scale = 2 ** he_cfg["global_scale_bits"]
    return ctx


def public_context(secret_ctx: ts.Context) -> ts.Context:
    """Strip the secret key -- this is what the SERVER actually holds. It
    can deserialize ciphertexts and add them, but cannot decrypt them
    (verified in the Stage 8c smoke test: attempting to decrypt with this
    context raises, exactly as it should).
    """
    return ts.context_from(secret_ctx.serialize(save_secret_key=False))


@dataclass
class HEOverhead:
    encrypt_seconds: float
    aggregate_seconds: float
    decrypt_seconds: float
    ciphertext_bytes: int
    plaintext_bytes: int


def encrypt_client_update(ctx: ts.Context, coef: np.ndarray, intercept: np.ndarray,
                          weight: float) -> tuple[bytes, float, int]:
    """Encrypt one client's size-weighted update (coef ++ intercept).

    The n_k/n weighting is applied in PLAINTEXT before encryption -- the
    same point at which the DP track applies its client-side noise --
    so that homomorphically summing the ciphertexts on the server yields
    the size-weighted average directly, with no plaintext arithmetic
    needed after decryption.
    """
    vec = np.concatenate([coef.ravel(), intercept.ravel()]) * weight
    t0 = time.perf_counter()
    enc = ts.ckks_vector(ctx, vec.tolist())
    serialized = enc.serialize()
    elapsed = time.perf_counter() - t0
    return serialized, elapsed, len(serialized)


def homomorphic_aggregate(server_ctx: ts.Context, serialized_updates: list[bytes]
                          ) -> tuple[bytes, float]:
    """Server-side homomorphic sum of the encrypted client updates.

    `server_ctx` is a PUBLIC context (no secret key) -- the server can
    perform this sum without ever being able to see an individual
    client's weights or the aggregate itself.
    """
    t0 = time.perf_counter()
    vectors = [ts.ckks_vector_from(server_ctx, s) for s in serialized_updates]
    total = vectors[0]
    for v in vectors[1:]:
        total = total + v
    serialized_sum = total.serialize()
    elapsed = time.perf_counter() - t0
    return serialized_sum, elapsed


def decrypt_aggregate(secret_ctx: ts.Context, serialized_sum: bytes, n_features: int
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """Decrypt the aggregated ciphertext into (coef, intercept). Only the
    secret-key holder -- never the server -- can perform this step.
    """
    t0 = time.perf_counter()
    enc = ts.ckks_vector_from(secret_ctx, serialized_sum)
    vec = np.array(enc.decrypt())
    elapsed = time.perf_counter() - t0
    return vec[:n_features], vec[n_features:], elapsed


def he_federated_round(secret_ctx: ts.Context, server_ctx: ts.Context,
                       global_coef: np.ndarray, global_intercept: np.ndarray,
                       X_clients: list, y_clients: list, sizes: list[int],
                       config: dict, seed: int, classes: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, HEOverhead]:
    """One FedAvg round aggregated via homomorphic encryption.

    Local training is identical to the plaintext track
    (`fedavg_lr.federated_round`) -- only the aggregation step differs --
    so any difference in outcome is attributable to HE itself, not to a
    different training procedure.
    """
    total = sum(sizes)
    n_features = global_coef.shape[0]

    serialized_updates = []
    encrypt_time = 0.0
    ciphertext_bytes = 0
    plaintext_bytes = 0

    for X, y, n in zip(X_clients, y_clients, sizes):
        local = build_local_lr(config, seed)
        set_weights(local, global_coef, global_intercept, classes)
        local.fit(X, y)

        weight = n / total
        serialized, t_enc, ct_bytes = encrypt_client_update(
            secret_ctx, local.coef_.ravel(), local.intercept_, weight
        )
        serialized_updates.append(serialized)
        encrypt_time += t_enc
        ciphertext_bytes += ct_bytes
        plaintext_bytes += (n_features + 1) * 8  # float64

    serialized_sum, agg_time = homomorphic_aggregate(server_ctx, serialized_updates)
    coef, intercept, dec_time = decrypt_aggregate(secret_ctx, serialized_sum, n_features)

    overhead = HEOverhead(
        encrypt_seconds=encrypt_time,
        aggregate_seconds=agg_time,
        decrypt_seconds=dec_time,
        ciphertext_bytes=ciphertext_bytes,
        plaintext_bytes=plaintext_bytes,
    )
    return coef, intercept, overhead


def train_fedavg_lr_he(X_clients: list, y_clients: list, sizes: list[int],
                       config: dict, seed: int
                       ) -> tuple[np.ndarray, np.ndarray, list, list[HEOverhead]]:
    """Run the full FedAvg loop for `federated.num_rounds` rounds, with
    every round's aggregation performed under homomorphic encryption.

    Returns the converged global (coef, intercept), the per-round weight
    history (for the convergence curve), and the per-round overhead
    measurements (for the cost analysis).
    """
    n_features = X_clients[0].shape[1]
    classes = np.array([0, 1])
    coef, intercept = np.zeros(n_features), np.zeros(1)

    secret_ctx = build_context(config)
    server_ctx = public_context(secret_ctx)

    num_rounds = config["federated"]["num_rounds"]
    history, overheads = [], []
    for r in range(num_rounds):
        coef, intercept, overhead = he_federated_round(
            secret_ctx, server_ctx, coef, intercept,
            X_clients, y_clients, sizes, config, seed + r, classes,
        )
        history.append((coef.copy(), intercept.copy()))
        overheads.append(overhead)

    return coef, intercept, history, overheads
