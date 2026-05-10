"""Matrix-Normal Wishart moment conversions."""

from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp

from bpc.config import BPCConfig, Array, CLIP_PSI_INV, JITTER, MIN_EIG
from bpc.posterior.natural_params import Eta
from bpc.posterior.posterior_state import LayerMoments
from bpc.utils.tensor_ops import eye, inv_spd, project_spd, sym


def natural_to_moment(eta: Eta, d_y: int, d_x: int) -> Tuple[Array, Array, Array, Array, Array]:
    V_inv = sym(eta.V_inv) + JITTER * eye(d_x, eta.V_inv.dtype)
    V = inv_spd(V_inv, JITTER)
    M = eta.MV @ V
    nu = eta.nu_shift + d_y - d_x + 1.0
    psi_inv_raw = sym(eta.S - eta.MV @ V @ eta.MV.T)
    psi_inv = project_spd(psi_inv_raw, MIN_EIG) if CLIP_PSI_INV else sym(psi_inv_raw + JITTER * eye(d_y, eta.S.dtype))
    Psi = inv_spd(psi_inv, JITTER)
    return M, V, Psi, nu, psi_inv_raw


def posterior_moments(params: Tuple[Eta, ...], layer_dims: Tuple[int, ...], cfg: BPCConfig) -> Tuple[LayerMoments, ...]:
    out: List[LayerMoments] = []
    for l, eta in enumerate(params):
        d_y = int(layer_dims[l + 1])
        d_x = int(layer_dims[l]) + 1
        M, V, Psi, nu, psi_inv_raw = natural_to_moment(eta, d_y, d_x)
        E_prec = sym(nu * Psi)
        E_prec_W = E_prec @ M
        E_Wt_prec_W = sym(M.T @ E_prec @ M + d_y * V)

        if cfg.precision_rescale == "unit_mean_precision":
            s = jnp.maximum(jnp.mean(jnp.diag(E_prec)), jnp.asarray(1e-12, dtype=E_prec.dtype))
            E_prec = E_prec / s
            E_prec_W = E_prec_W / s
            E_Wt_prec_W = E_Wt_prec_W / s
        elif cfg.precision_rescale != "none":
            raise ValueError(f"Unknown precision_rescale={cfg.precision_rescale}")

        out.append(LayerMoments(
            M,
            V,
            E_prec,
            E_prec_W,
            E_Wt_prec_W,
            jnp.min(jnp.linalg.eigvalsh(sym(psi_inv_raw))),
        ))
    return tuple(out)
