"""Hidden-state initialization routines."""

from __future__ import annotations

from typing import List, Tuple

import jax
import jax.numpy as jnp

from bpc.activations import activation
from bpc.config import BPCConfig, Array
from bpc.distributions.matrix_normal_wishart import posterior_moments
from bpc.posterior.natural_params import Eta
from bpc.utils.tensor_ops import augment


def feedforward_init(params: Tuple[Eta, ...], layer_dims: Tuple[int, ...], x: Array, cfg: BPCConfig) -> Tuple[Array, ...]:
    z = x
    hidden: List[Array] = []
    moms = posterior_moments(params, layer_dims, cfg)
    for l, mom in enumerate(moms[:-1]):
        z = augment(activation(z, l, cfg.input_activation)) @ mom.M.T
        hidden.append(z)
    return tuple(hidden)


def hidden_init(params: Tuple[Eta, ...], layer_dims: Tuple[int, ...], x: Array, cfg: BPCConfig, key: Array) -> Tuple[Array, ...]:
    if cfg.hidden_init == "feedforward":
        return feedforward_init(params, layer_dims, x, cfg)
    if cfg.hidden_init == "zeros":
        return tuple(jnp.zeros((x.shape[0], layer_dims[l]), dtype=x.dtype) for l in range(1, len(layer_dims) - 1))
    if cfg.hidden_init == "random_normal":
        keys = jax.random.split(key, len(layer_dims) - 2)
        return tuple(cfg.hidden_init_std * jax.random.normal(keys[l - 1], (x.shape[0], layer_dims[l]), dtype=x.dtype) for l in range(1, len(layer_dims) - 1))
    raise ValueError(f"Unknown hidden_init={cfg.hidden_init}")
