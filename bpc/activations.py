"""Activation functions used by the faithful BPC implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bpc.config import Array


def activation(z: Array, layer_index: int, input_activation: str) -> Array:
    if layer_index == 0:
        return jax.nn.relu(z) if input_activation == "relu" else z
    return jax.nn.relu(z)


def activation_grad(z: Array, layer_index: int, input_activation: str) -> Array:
    if layer_index == 0:
        return (z > 0).astype(z.dtype) if input_activation == "relu" else jnp.ones_like(z)
    return (z > 0).astype(z.dtype)
