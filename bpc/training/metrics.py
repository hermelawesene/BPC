"""Prediction metrics and posterior diagnostics."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from bpc.config import BPCConfig, Array
from bpc.distributions.matrix_normal_wishart import natural_to_moment
from bpc.layers.predictive_network import predict
from bpc.posterior.natural_params import Eta
from bpc.utils.tensor_ops import sym


def classification_accuracy(params: Tuple[Eta, ...], layer_dims: Tuple[int, ...], x: Array, y: Array, cfg: BPCConfig) -> float:
    pred_fn = jax.jit(lambda p, xb: predict(p, layer_dims, xb, cfg))
    total = int(x.shape[0])
    correct = 0
    for start in range(0, total, cfg.eval_batch_size):
        xb = x[start:start + cfg.eval_batch_size]
        yb = y[start:start + cfg.eval_batch_size]
        logits = pred_fn(params, xb)
        correct += int(jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(yb, axis=-1)))
    return correct / max(total, 1)


def per_class_accuracy(params: Tuple[Eta, ...], layer_dims: Tuple[int, ...], x: Array, y: Array, cfg: BPCConfig) -> Dict[int, float]:
    pred_fn = jax.jit(lambda p, xb: predict(p, layer_dims, xb, cfg))
    total = int(x.shape[0])
    n_classes = int(y.shape[-1])
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)
    for start in range(0, total, cfg.eval_batch_size):
        xb = x[start:start + cfg.eval_batch_size]
        yb = y[start:start + cfg.eval_batch_size]
        logits = pred_fn(params, xb)
        pred = np.asarray(jnp.argmax(logits, axis=-1))
        true = np.asarray(jnp.argmax(yb, axis=-1))
        for c in range(n_classes):
            mask = true == c
            total_per_class[c] += int(mask.sum())
            correct_per_class[c] += int(((pred == c) & mask).sum())
    return {
        c: float(correct_per_class[c] / max(total_per_class[c], 1))
        for c in range(n_classes)
    }


def parameter_diagnostics(params: Tuple[Eta, ...], layer_dims: Tuple[int, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for l, eta in enumerate(params):
        d_y = int(layer_dims[l + 1])
        d_x = int(layer_dims[l]) + 1
        M, V, Psi, nu, psi_inv_raw = natural_to_moment(eta, d_y, d_x)
        out[f"M{l}_norm"] = float(jnp.linalg.norm(M))
        out[f"nu{l}"] = float(nu)
        out[f"Vinv{l}_eig_min"] = float(jnp.min(jnp.linalg.eigvalsh(sym(eta.V_inv))))
        out[f"PsiInvRaw{l}_eig_min"] = float(jnp.min(jnp.linalg.eigvalsh(sym(psi_inv_raw))))
        out[f"V{l}_diag_min"] = float(jnp.min(jnp.diag(V)))
        out[f"V{l}_diag_max"] = float(jnp.max(jnp.diag(V)))
        out[f"Psi{l}_diag_min"] = float(jnp.min(jnp.diag(Psi)))
        out[f"Psi{l}_diag_max"] = float(jnp.max(jnp.diag(Psi)))
    return out


def has_bad_values(params: Tuple[Eta, ...]) -> bool:
    leaves = jax.tree.leaves(params)
    return any(bool(jnp.any(~jnp.isfinite(x))) for x in leaves)
