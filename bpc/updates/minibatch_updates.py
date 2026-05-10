"""Kappa and mini-batch scaling schedules."""

from __future__ import annotations

from bpc.config import BPCConfig


def compute_kappa(step: int, epoch: int, cfg: BPCConfig) -> float:
    if cfg.kappa_burnin_value is not None and epoch <= cfg.kappa_burnin_epochs:
        val = float(cfg.kappa_burnin_value)
    else:
        clock_value = epoch if cfg.kappa_clock == "epoch" else step
        val = cfg.kappa_multiplier * ((cfg.kappa_delay + max(1, clock_value)) ** (-cfg.kappa_exponent))
    if cfg.kappa_max is not None:
        val = min(val, cfg.kappa_max)
    if cfg.kappa_min is not None:
        val = max(val, cfg.kappa_min)
    return float(val)


def compute_scale(n_total: int, batch_size: int, cfg: BPCConfig) -> float:
    if cfg.update_mode == "batch_local":
        scale = 1.0
    elif cfg.update_mode == "svi_scaled":
        scale = n_total / float(batch_size)
    elif cfg.update_mode == "svi_power":
        scale = (n_total / float(batch_size)) ** cfg.scale_power
    else:
        scale = 1.0
    scale *= cfg.scale_multiplier
    if cfg.scale_cap is not None:
        scale = min(scale, cfg.scale_cap)
    return float(scale)
