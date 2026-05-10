"""Prediction through the expected-weight BPC network."""

from __future__ import annotations

from typing import Tuple

from bpc.activations import activation
from bpc.config import BPCConfig, Array
from bpc.distributions.matrix_normal_wishart import posterior_moments
from bpc.posterior.natural_params import Eta
from bpc.utils.tensor_ops import augment


def predict(params: Tuple[Eta, ...], layer_dims: Tuple[int, ...], x: Array, cfg: BPCConfig) -> Array:
    moms = posterior_moments(params, layer_dims, cfg)
    z = x
    for l, mom in enumerate(moms):
        z = augment(activation(z, l, cfg.input_activation)) @ mom.M.T
    return z
