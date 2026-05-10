"""Matrix-Normal Wishart prior and initial posterior construction."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp

from bpc.config import Array, DTYPE
from bpc.posterior.natural_params import Eta
from bpc.utils.tensor_ops import eye, sym


def init_prior(
    layer_dims: Sequence[int],
    v0: float = 10.0,
    psi0: float = 1000.0,
    nu0_extra: float = 2.0,
) -> Tuple[Eta, ...]:
    priors: List[Eta] = []
    for l in range(len(layer_dims) - 1):
        d_y = int(layer_dims[l + 1])
        d_x = int(layer_dims[l]) + 1
        V_inv = (1.0 / v0) * eye(d_x)
        MV = jnp.zeros((d_y, d_x), dtype=DTYPE)
        S = (1.0 / psi0) * eye(d_y)
        nu = d_y + float(nu0_extra)
        nu_shift = jnp.asarray(nu - d_y + d_x - 1.0, dtype=DTYPE)
        priors.append(Eta(V_inv, MV, S, nu_shift))
    return tuple(priors)


def init_posterior(
    layer_dims: Sequence[int],
    key: Array,
    v0: float = 10.0,
    psi0: float = 1000.0,
    nu0_extra: float = 2.0,
) -> Tuple[Eta, ...]:
    priors = init_prior(layer_dims, v0=v0, psi0=psi0, nu0_extra=nu0_extra)
    post: List[Eta] = []
    for l, eta0 in enumerate(priors):
        d_y = int(layer_dims[l + 1])
        d_x_no_bias = int(layer_dims[l])
        d_x = d_x_no_bias + 1
        key, sub = jax.random.split(key)
        bound = 1.0 / jnp.sqrt(jnp.asarray(d_x_no_bias, dtype=DTYPE))
        M = jax.random.uniform(sub, (d_y, d_x), minval=-bound, maxval=bound, dtype=DTYPE)
        MV = M @ eta0.V_inv
        S = eta0.S + MV @ M.T
        post.append(Eta(sym(eta0.V_inv), MV, sym(S), eta0.nu_shift))
    return tuple(post)
