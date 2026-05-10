"""Shape and configuration guardrails used at module boundaries."""

from __future__ import annotations

from typing import Tuple

from bpc.config import BPCConfig


def validate_layer_dims(layer_dims: Tuple[int, ...]) -> None:
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must include at least input and output dimensions.")
    if any(int(d) <= 0 for d in layer_dims):
        raise ValueError(f"layer_dims must be positive, got {layer_dims}.")


def validate_config(cfg: BPCConfig) -> None:
    if cfg.update_mode not in {"batch_local", "svi_scaled", "svi_power", "epoch_exact", "epoch_ema", "online_additive"}:
        raise ValueError(f"Unknown update_mode={cfg.update_mode}")
    if cfg.hidden_init not in {"feedforward", "zeros", "random_normal"}:
        raise ValueError(f"Unknown hidden_init={cfg.hidden_init}")
    if cfg.input_activation not in {"identity", "relu"}:
        raise ValueError(f"Unknown input_activation={cfg.input_activation}")
    if cfg.normalize not in {"zero_one", "standardize_pixel", "standardize_scalar", "centered_m11"}:
        raise ValueError(f"Unknown normalize={cfg.normalize}")
