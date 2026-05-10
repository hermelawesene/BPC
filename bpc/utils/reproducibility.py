"""Seed helpers."""

from __future__ import annotations

import jax
import numpy as np


def make_jax_key(seed: int):
    return jax.random.PRNGKey(seed)


def make_numpy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
