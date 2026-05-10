"""Posterior natural-parameter updates and M-step diagnostics."""

from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp

from bpc.config import Array
from bpc.distributions.matrix_normal_wishart import natural_to_moment
from bpc.posterior.natural_params import Eta
from bpc.posterior.posterior_state import MStepDiagnostics, Stats, StatsDiagnostics
from bpc.utils.tensor_ops import sym


def stats_to_eta(prior: Tuple[Eta, ...], stats: Tuple[Stats, ...], scale: Array) -> Tuple[Eta, ...]:
    out: List[Eta] = []
    for eta0, st in zip(prior, stats):
        out.append(Eta(
            sym(eta0.V_inv + scale * st.T1),
            eta0.MV + scale * st.T2,
            sym(eta0.S + scale * st.T3),
            eta0.nu_shift + scale * st.T4,
        ))
    return tuple(out)


def interpolate_eta(a: Tuple[Eta, ...], b: Tuple[Eta, ...], kappa: Array) -> Tuple[Eta, ...]:
    out: List[Eta] = []
    for ea, eb in zip(a, b):
        out.append(Eta(
            sym((1.0 - kappa) * ea.V_inv + kappa * eb.V_inv),
            (1.0 - kappa) * ea.MV + kappa * eb.MV,
            sym((1.0 - kappa) * ea.S + kappa * eb.S),
            (1.0 - kappa) * ea.nu_shift + kappa * eb.nu_shift,
        ))
    return tuple(out)


def additive_eta(current: Tuple[Eta, ...], stats: Tuple[Stats, ...], kappa: Array) -> Tuple[Eta, ...]:
    out: List[Eta] = []
    for eta, st in zip(current, stats):
        out.append(Eta(
            sym(eta.V_inv + kappa * st.T1),
            eta.MV + kappa * st.T2,
            sym(eta.S + kappa * st.T3),
            eta.nu_shift + kappa * st.T4,
        ))
    return tuple(out)


def compute_stats_diagnostics(stats: Tuple[Stats, ...]) -> StatsDiagnostics:
    return StatsDiagnostics(
        T1_norm=jnp.stack([jnp.linalg.norm(st.T1) for st in stats]),
        T2_norm=jnp.stack([jnp.linalg.norm(st.T2) for st in stats]),
        T3_norm=jnp.stack([jnp.linalg.norm(st.T3) for st in stats]),
        T4=jnp.stack([st.T4 for st in stats]),
    )


def compute_mstep_diagnostics(
    old: Tuple[Eta, ...],
    new: Tuple[Eta, ...],
    layer_dims: Tuple[int, ...],
) -> MStepDiagnostics:
    eta_delta_Vinv = jnp.stack([jnp.linalg.norm(n.V_inv - o.V_inv) for o, n in zip(old, new)])
    eta_delta_MV = jnp.stack([jnp.linalg.norm(n.MV - o.MV) for o, n in zip(old, new)])
    eta_delta_S = jnp.stack([jnp.linalg.norm(n.S - o.S) for o, n in zip(old, new)])
    eta_delta_nu = jnp.stack([jnp.abs(n.nu_shift - o.nu_shift) for o, n in zip(old, new)])

    new_M_norms = []
    new_psi_inv_raw_min = []
    for l, eta in enumerate(new):
        d_y = int(layer_dims[l + 1])
        d_x = int(layer_dims[l]) + 1
        M, V, Psi, nu, psi_inv_raw = natural_to_moment(eta, d_y, d_x)
        new_M_norms.append(jnp.linalg.norm(M))
        new_psi_inv_raw_min.append(jnp.min(jnp.linalg.eigvalsh(sym(psi_inv_raw))))

    return MStepDiagnostics(
        eta_delta_Vinv=eta_delta_Vinv,
        eta_delta_MV=eta_delta_MV,
        eta_delta_S=eta_delta_S,
        eta_delta_nu=eta_delta_nu,
        new_M_norm=jnp.stack(new_M_norms),
        new_psi_inv_raw_min=jnp.stack(new_psi_inv_raw_min),
    )
