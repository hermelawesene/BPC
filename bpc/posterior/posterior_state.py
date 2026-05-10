"""State and diagnostic containers for BPC posterior and latent inference."""

from __future__ import annotations

from typing import NamedTuple

from bpc.config import Array


class Stats(NamedTuple):
    T1: Array
    T2: Array
    T3: Array
    T4: Array


class LatentDiagnostics(NamedTuple):
    grad_norm_per_step: Array
    latent_norm_per_step: Array
    energy_per_step: Array
    init_grad_norm: Array
    final_grad_norm: Array
    M_norm: Array
    Vinv_fro: Array
    psi_inv_raw_min: Array
    E_prec_diag_mean: Array


class StatsDiagnostics(NamedTuple):
    T1_norm: Array
    T2_norm: Array
    T3_norm: Array
    T4: Array


class MStepDiagnostics(NamedTuple):
    eta_delta_Vinv: Array
    eta_delta_MV: Array
    eta_delta_S: Array
    eta_delta_nu: Array
    new_M_norm: Array
    new_psi_inv_raw_min: Array


class LayerMoments(NamedTuple):
    M: Array
    V: Array
    E_prec: Array
    E_prec_W: Array
    E_Wt_prec_W: Array
    psi_inv_raw_min: Array
