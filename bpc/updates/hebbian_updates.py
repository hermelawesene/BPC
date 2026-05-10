"""Sufficient-statistics accumulation for the BPC M-step."""

from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp

from bpc.activations import activation
from bpc.config import BPCConfig, Array, DTYPE
from bpc.posterior.posterior_state import Stats
from bpc.utils.tensor_ops import augment, sym


def zero_stats(layer_dims: Tuple[int, ...]) -> Tuple[Stats, ...]:
    stats: List[Stats] = []
    for l in range(len(layer_dims) - 1):
        d_x = int(layer_dims[l]) + 1
        d_y = int(layer_dims[l + 1])
        stats.append(Stats(
            jnp.zeros((d_x, d_x), dtype=DTYPE),
            jnp.zeros((d_y, d_x), dtype=DTYPE),
            jnp.zeros((d_y, d_y), dtype=DTYPE),
            jnp.asarray(0.0, dtype=DTYPE),
        ))
    return tuple(stats)


def add_stats(a: Tuple[Stats, ...], b: Tuple[Stats, ...]) -> Tuple[Stats, ...]:
    return tuple(Stats(sym(x.T1 + y.T1), x.T2 + y.T2, sym(x.T3 + y.T3), x.T4 + y.T4) for x, y in zip(a, b))


def sufficient_statistics(
    hidden: Tuple[Array, ...],
    x: Array,
    y: Array,
    layer_dims: Tuple[int, ...],
    cfg: BPCConfig,
) -> Tuple[Stats, ...]:
    z_all = (x,) + tuple(hidden) + (y,)
    stats: List[Stats] = []
    for l in range(len(layer_dims) - 1):
        f_aug = augment(activation(z_all[l], l, cfg.input_activation))
        z_out = z_all[l + 1]
        stats.append(Stats(
            sym(f_aug.T @ f_aug),
            z_out.T @ f_aug,
            sym(z_out.T @ z_out),
            jnp.asarray(z_out.shape[0], dtype=DTYPE),
        ))
    return tuple(stats)
