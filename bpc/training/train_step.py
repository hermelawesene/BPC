"""JIT-compiled training-step factories used by the high-level trainers."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from bpc.config import BPCConfig, Array, DTYPE
from bpc.distributions.matrix_normal_wishart import posterior_moments
from bpc.inference.hidden_state_inference import adam_latent_inference_diag
from bpc.posterior.natural_params import Eta
from bpc.posterior.posterior_state import Stats
from bpc.priors.hidden_state_prior import hidden_init
from bpc.updates.hebbian_updates import sufficient_statistics
from bpc.updates.posterior_updates import (
    additive_eta,
    compute_mstep_diagnostics,
    compute_stats_diagnostics,
    interpolate_eta,
    stats_to_eta,
)


def make_two_moons_epoch_with_diag(
    cfg: BPCConfig,
    layer_dims: Tuple[int, ...],
    prior: Tuple[Eta, ...],
    x_train: Array,
    y_train: Array,
):
    @jax.jit
    def one_epoch_with_diag(params, key):
        moms = posterior_moments(params, layer_dims, cfg)
        h0 = hidden_init(params, layer_dims, x_train, cfg, key)
        h, ldiag = adam_latent_inference_diag(h0, moms, x_train, y_train, cfg)
        st = sufficient_statistics(h, x_train, y_train, layer_dims, cfg)
        sdiag = compute_stats_diagnostics(st)
        new_params = stats_to_eta(prior, st, jnp.asarray(1.0, dtype=DTYPE))
        mdiag = compute_mstep_diagnostics(params, new_params, layer_dims)
        return new_params, ldiag, sdiag, mdiag

    return one_epoch_with_diag


def make_infer_stats_with_diag(
    cfg: BPCConfig,
    warmup_cfg: BPCConfig,
    layer_dims: Tuple[int, ...],
) -> Callable[[Tuple[Eta, ...], Array, Array, Array, int], Tuple[Tuple[Stats, ...], object, object]]:
    @jax.jit
    def infer_stats_main_with_diag(params, xb, yb, key):
        moms = posterior_moments(params, layer_dims, cfg)
        h0 = hidden_init(params, layer_dims, xb, cfg, key)
        h, ldiag = adam_latent_inference_diag(h0, moms, xb, yb, cfg)
        st = sufficient_statistics(h, xb, yb, layer_dims, cfg)
        sdiag = compute_stats_diagnostics(st)
        return st, ldiag, sdiag

    @jax.jit
    def infer_stats_warmup_with_diag(params, xb, yb, key):
        moms = posterior_moments(params, layer_dims, warmup_cfg)
        h0 = hidden_init(params, layer_dims, xb, warmup_cfg, key)
        h, ldiag = adam_latent_inference_diag(h0, moms, xb, yb, warmup_cfg)
        st = sufficient_statistics(h, xb, yb, layer_dims, warmup_cfg)
        sdiag = compute_stats_diagnostics(st)
        return st, ldiag, sdiag

    def infer_stats_for_epoch(params, xb, yb, key, epoch_index: int):
        if cfg.warmup_epochs > 0 and epoch_index <= cfg.warmup_epochs:
            return infer_stats_warmup_with_diag(params, xb, yb, key)
        return infer_stats_main_with_diag(params, xb, yb, key)

    return infer_stats_for_epoch


def make_apply_target_with_diag(
    cfg: BPCConfig,
    layer_dims: Tuple[int, ...],
    prior: Tuple[Eta, ...],
):
    @jax.jit
    def apply_target_with_diag(current, st, kappa, scale):
        if cfg.update_mode == "online_additive":
            new_params = additive_eta(current, st, kappa)
        else:
            target = stats_to_eta(prior, st, scale)
            new_params = interpolate_eta(current, target, kappa)
        mdiag = compute_mstep_diagnostics(current, new_params, layer_dims)
        return new_params, mdiag

    return apply_target_with_diag


def make_apply_exact_or_ema_with_diag(
    cfg: BPCConfig,
    layer_dims: Tuple[int, ...],
    prior: Tuple[Eta, ...],
):
    @jax.jit
    def apply_exact_or_ema_with_diag(current, total_stats, kappa):
        target = stats_to_eta(prior, total_stats, jnp.asarray(1.0, dtype=DTYPE))
        if cfg.update_mode == "epoch_exact":
            new_params = target
        else:
            new_params = interpolate_eta(current, target, kappa)
        mdiag = compute_mstep_diagnostics(current, new_params, layer_dims)
        return new_params, mdiag

    return apply_exact_or_ema_with_diag
