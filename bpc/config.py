"""Configuration and process-wide JAX setup for the faithful BPC refactor."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, Optional


RUN_MODE = "mnist_single"
SELECTED_PRESET = "paper_svi_delay5000_cap005_3hidden_zeroone"
SWEEP_PRESETS = [
    "paper_literal_batch_local",
    "paper_svi_scaled",
    "paper_svi_delay5000_cap005_3hidden_zeroone",
    "paper_svi_delay10000_cap005_3hidden_zeroone",
    "stabilized_svi_delay5000_cap005_2hidden_std",
    "svi_scaled_burnin001_2hidden_std",
    "diagnostic_stable_epoch_exact",
    "relaxed_prior_svi_2hidden_std",
]

ENABLE_X64 = True
SAVE_DIR = "/content/bpc_faithful_runs"
PLOT_CURVES = True
LOGGING_MODE = "verbose"

SEED = 0
MNIST_EPOCHS = 20
BATCH_SIZE = 128
HIDDEN = 400
LATENT_STEPS = 10
LATENT_LR = 0.01
KAPPA_EXPONENT = 0.25
MNIST_SOURCE = "keras"
MNIST_NPZ = None

TWO_MOONS_EPOCHS = 80
TWO_MOONS_HIDDEN = 100
TWO_MOONS_NOISE = 0.10

JITTER = 1e-7
MIN_EIG = 1e-10
CLIP_PSI_INV = True
GRAD_NORM_CLIP = None
STOP_ON_NAN = True
REPORT_TRAIN_SUBSET_ACC = True
TRAIN_SUBSET_SIZE = 10000

if ENABLE_X64:
    os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


jax.config.update("jax_default_prng_impl", "threefry2x32")
if ENABLE_X64:
    jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray
DTYPE = jnp.float64 if ENABLE_X64 else jnp.float32


@dataclass(frozen=True)
class BPCConfig:
    name: str
    seed: int = SEED
    epochs: int = MNIST_EPOCHS
    batch_size: int = BATCH_SIZE
    hidden: int = HIDDEN
    hidden_layers: int = 3

    normalize: str = "zero_one"
    mnist_source: str = MNIST_SOURCE
    mnist_npz: Optional[str] = MNIST_NPZ

    v0: float = 10.0
    psi0: float = 1000.0
    nu0_extra: float = 2.0

    latent_steps: int = LATENT_STEPS
    latent_lr: float = LATENT_LR
    hidden_init: str = "feedforward"
    hidden_init_std: float = 0.01
    input_activation: str = "identity"

    warmup_epochs: int = 0
    warmup_latent_steps: Optional[int] = None
    warmup_latent_lr: Optional[float] = None
    warmup_hidden_init: Optional[str] = None

    update_mode: str = "batch_local"
    scale_power: float = 1.0
    scale_multiplier: float = 1.0
    scale_cap: Optional[float] = None

    kappa_exponent: float = KAPPA_EXPONENT
    kappa_delay: float = 0.0
    kappa_multiplier: float = 1.0
    kappa_min: Optional[float] = None
    kappa_max: Optional[float] = None
    kappa_clock: str = "batch"

    kappa_burnin_epochs: int = 0
    kappa_burnin_value: Optional[float] = None

    precision_rescale: str = "none"

    eval_batch_size: int = 4096
    keep_best: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for the diagnostic logging machinery."""

    enabled: bool = True
    save_dir: Optional[str] = None
    run_id_prefix: str = ""

    batch_csv: bool = True
    verbose_jsonl: bool = True
    adam_trace_jsonl: bool = True

    per_class_acc: bool = True
    parameter_diag_every_epoch: bool = True
    log_init_posterior: bool = True
    log_data_stats: bool = True
    write_manifest: bool = True
    extra_plots: bool = True
    plot_dpi: int = 120

    t_norm_alarm: float = 1e8
    psi_alarm: float = 1e-12
    eta_jump_alarm: float = 1e6
    grad_alarm: float = 1e6


VERBOSE_LOGGING = LoggingConfig()
LITE_LOGGING = LoggingConfig(
    batch_csv=False,
    verbose_jsonl=False,
    adam_trace_jsonl=False,
    extra_plots=False,
)
LOGGING = VERBOSE_LOGGING if LOGGING_MODE == "verbose" else LITE_LOGGING


def make_presets() -> Dict[str, BPCConfig]:
    base = BPCConfig(name="base")
    return {
        "paper_literal_batch_local": replace(
            base,
            name="paper_literal_batch_local",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="batch_local",
            kappa_delay=0.0,
            kappa_multiplier=1.0,
            kappa_max=None,
            scale_power=0.0,
        ),
        "paper_svi_scaled": replace(
            base,
            name="paper_svi_scaled",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="svi_scaled",
            kappa_delay=0.0,
            kappa_multiplier=1.0,
            kappa_max=None,
            scale_power=1.0,
        ),
        "paper_svi_delay5000_cap005_3hidden_zeroone": replace(
            base,
            name="paper_svi_delay5000_cap005_3hidden_zeroone",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="svi_scaled",
            scale_power=1.0,
            kappa_delay=5000.0,
            kappa_max=0.05,
            hidden_init="zeros",
        ),
        "paper_svi_delay10000_cap005_3hidden_zeroone": replace(
            base,
            name="paper_svi_delay10000_cap005_3hidden_zeroone",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="svi_scaled",
            scale_power=1.0,
            kappa_delay=10000.0,
            kappa_max=0.05,
            hidden_init="zeros",
        ),
        "stabilized_svi_delay5000_cap005_2hidden_std": replace(
            base,
            name="stabilized_svi_delay5000_cap005_2hidden_std",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="svi_scaled",
            scale_power=1.0,
            kappa_delay=5000.0,
            kappa_max=0.05,
            hidden_init="zeros",
            latent_steps=20,
            latent_lr=0.005,
        ),
        "svi_scaled_burnin001_2hidden_std": replace(
            base,
            name="svi_scaled_burnin001_2hidden_std",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="svi_scaled",
            scale_power=1.0,
            kappa_delay=5000.0,
            kappa_max=0.05,
            kappa_burnin_epochs=5,
            kappa_burnin_value=0.01,
            hidden_init="zeros",
            latent_steps=20,
            latent_lr=0.005,
        ),
        "svi_power_half_delay5000_cap005_2hidden_std": replace(
            base,
            name="svi_power_half_delay5000_cap005_2hidden_std",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="svi_power",
            scale_power=0.50,
            kappa_delay=5000.0,
            kappa_max=0.05,
            hidden_init="zeros",
            latent_steps=20,
            latent_lr=0.005,
        ),
        "svi_power_quarter_delay5000_cap005_2hidden_std": replace(
            base,
            name="svi_power_quarter_delay5000_cap005_2hidden_std",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="svi_power",
            scale_power=0.25,
            kappa_delay=5000.0,
            kappa_max=0.05,
            hidden_init="zeros",
            latent_steps=20,
            latent_lr=0.005,
        ),
        "diagnostic_stable_epoch_exact": replace(
            base,
            name="diagnostic_stable_epoch_exact",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="epoch_exact",
            kappa_clock="epoch",
            kappa_delay=10.0,
            kappa_max=0.5,
            latent_steps=20,
            latent_lr=0.005,
            hidden_init="zeros",
        ),
        "paper_svi_delayed_halfscale": replace(
            base,
            name="paper_svi_delayed_halfscale",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="svi_power",
            scale_power=0.50,
            scale_multiplier=1.0,
            scale_cap=None,
            kappa_delay=1000.0,
            kappa_multiplier=1.0,
            kappa_max=0.20,
        ),
        "paper_svi_delayed_quarterscale": replace(
            base,
            name="paper_svi_delayed_quarterscale",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="svi_power",
            scale_power=0.25,
            kappa_delay=1000.0,
            kappa_multiplier=1.0,
            kappa_max=0.20,
        ),
        "paper_batchlocal_delayed_standardized": replace(
            base,
            name="paper_batchlocal_delayed_standardized",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="batch_local",
            kappa_delay=1000.0,
            kappa_max=0.15,
        ),
        "epoch_ema_standardized_2hidden": replace(
            base,
            name="epoch_ema_standardized_2hidden",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="epoch_ema",
            kappa_clock="epoch",
            kappa_delay=10.0,
            kappa_max=0.50,
        ),
        "epoch_exact_zeroone_3hidden": replace(
            base,
            name="epoch_exact_zeroone_3hidden",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="epoch_exact",
        ),
        "relaxed_prior_svi_2hidden_std": replace(
            base,
            name="relaxed_prior_svi_2hidden_std",
            hidden_layers=2,
            normalize="standardize_pixel",
            update_mode="svi_scaled",
            scale_power=1.0,
            kappa_delay=5000.0,
            kappa_max=0.05,
            hidden_init="zeros",
            latent_steps=20,
            latent_lr=0.005,
            psi0=100.0,
            nu0_extra=10.0,
        ),
        "mnist_feedforward_stable": replace(
            base,
            name="mnist_feedforward_stable",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="svi_power",     # gentler than svi_scaled
            scale_power=0.5,              # scale = sqrt(N/B) ≈ 21.7 instead of 469
            kappa_delay=1000.0,
            kappa_max=0.15,               # allow larger kappa than 0.05
            hidden_init="feedforward",   # warm start → alive gradients
            latent_steps=20,
            latent_lr=0.005,
        ),
        "mnist_stable_v2": replace(
            base,
            name="mnist_stable_v2",
            hidden_layers=3,
            normalize="zero_one",
            update_mode="svi_power",        # Gentler than batch_local
            scale_power=0.4,                # Even softer scaling
            kappa_delay=1500.0,             # Give it time to settle
            kappa_max=0.08,                 # Conservative max learning rate
            hidden_init="feedforward",
            latent_steps=25,
            latent_lr=0.004,
            warmup_epochs=2,                # Extra gentle start
            warmup_latent_steps=30,
            warmup_latent_lr=0.01,
        ),
        "best_stable_mnist": replace(
            base,
            name="best_stable_mnist",
            hidden_layers=3,
            normalize="zero_one",
            
            # --- Core Stable Settings ---
            hidden_init="feedforward",           # Very important
            latent_steps=25,
            latent_lr=0.004,
            
            update_mode="svi_power",             # Gentler than batch_local
            scale_power=0.35,                    # Quite soft scaling (~ sqrt(sqrt(N/B)))
            kappa_delay=2000.0,                  # Give early epochs time to stabilize
            kappa_max=0.09,                      # Conservative but not too weak
            kappa_multiplier=1.0,
            
            # Optional extra safety
            warmup_epochs=3,
            warmup_latent_steps=30,
            warmup_latent_lr=0.008,
            warmup_hidden_init="feedforward",
        ),
        "mnist_strong_stable_v3": replace(
            base,
            name="mnist_strong_stable_v3",
            hidden_layers=3,
            normalize="zero_one",

            hidden_init="feedforward",
            latent_steps=30,           # More inference steps
            latent_lr=0.003,           # Slower, more careful inference

            update_mode="svi_power",
            scale_power=0.25,          # Much gentler scaling
            kappa_delay=3000.0,        # Delay strong updates longer
            kappa_max=0.06,            # Quite conservative
            kappa_multiplier=0.9,

            warmup_epochs=5,           # Longer warmup
            warmup_latent_steps=40,
            warmup_latent_lr=0.01,

            precision_rescale="unit_mean_precision",   # Normalize precision across layers
        ),
        "mnist_ultra_stable_v4": replace(
            base,
            name="mnist_ultra_stable_v4",
            hidden_layers=3,
            normalize="zero_one",

            hidden_init="feedforward",
            latent_steps=35,                    # More careful inference
            latent_lr=0.0025,                   # Even slower inference LR

            update_mode="svi_power",
            scale_power=0.20,                   # Very gentle scaling
            kappa_delay=4000.0,                 # Delay strong updates longer
            kappa_max=0.055,                    # Very conservative
            kappa_multiplier=0.85,

            warmup_epochs=6,
            warmup_latent_steps=40,
            warmup_latent_lr=0.009,

            precision_rescale="unit_mean_precision",   # Normalize precision per layer
        ),
        "mnist_epoch_stable": replace(
            base,
            name="mnist_epoch_stable",
            hidden_layers=2,                    # Start with 2 layers (much easier)
            normalize="zero_one",

            hidden_init="feedforward",
            latent_steps=30,
            latent_lr=0.003,

            update_mode="epoch_exact",          # ← Big change: Full epoch update (closer to paper)
            # No need for scale_power or high kappa in epoch mode

            kappa_delay=0.0,
            kappa_max=1.0,                      # Full update at end of epoch
            kappa_clock="epoch",

            warmup_epochs=5,
            warmup_latent_steps=40,
            warmup_latent_lr=0.008,

            precision_rescale="unit_mean_precision",
        ),
        "mnist_best_hybrid_v5": replace(
            base,# bestttt so far
            name="mnist_best_hybrid_v5",
            hidden_layers=2,                    
            normalize="zero_one",
            hidden=400,

            hidden_init="feedforward",
            latent_steps=30,
            latent_lr=0.003,

            update_mode="svi_power",
            scale_power=0.25,
            kappa_delay=2500.0,
            kappa_max=0.07,

            warmup_epochs=5,
            warmup_latent_steps=35,
            warmup_latent_lr=0.009,

            precision_rescale="unit_mean_precision",
            psi0=200.0,                         
            nu0_extra=5.0,
        ),
        "mnist_best_hybrid_vv6": replace(
            base,
            name="mnist_best_hybrid_vv6",
            hidden=400,                    # Keep wide network
            hidden_layers=2,
            normalize="zero_one",

            hidden_init="feedforward",
            latent_steps=40,               # More inference time
            latent_lr=0.002,               # Slower, more stable inference

            update_mode="svi_power",
            scale_power=0.20,              # Gentler scaling
            kappa_delay=3500.0,            # Delay strong updates longer
            kappa_max=0.055,               # More conservative after initial learning

            warmup_epochs=8,               # Longer warmup
            warmup_latent_steps=50,
            warmup_latent_lr=0.008,

            precision_rescale="unit_mean_precision",
            psi0=300.0,                    # Slightly stronger prior
            nu0_extra=6.0,
        ),
        "mnist_hybrid_v6": replace(
            base,
            name="mnist_hybrid_v6",
            hidden_layers=3,                    # Go back to 3 layers now that we have stability
            normalize="zero_one",

            hidden_init="feedforward",
            latent_steps=30,
            latent_lr=0.0028,

            update_mode="svi_power",
            scale_power=0.22,                   # Slightly gentler
            kappa_delay=3000.0,
            kappa_max=0.065,

            warmup_epochs=6,
            warmup_latent_steps=40,
            warmup_latent_lr=0.008,

            precision_rescale="unit_mean_precision",
            psi0=250.0,                         # Slightly stronger prior than last
            nu0_extra=4.0,
        ),
        "mnist_top_attempt": replace(
            base,
            name="mnist_top_attempt",
            hidden_layers=2,                    # Stability first
            normalize="zero_one",

            hidden_init="feedforward",
            latent_steps=35,
            latent_lr=0.0025,

            update_mode="svi_power",
            scale_power=0.18,                   # Very gentle
            kappa_delay=3500.0,
            kappa_max=0.05,                     # Conservative

            warmup_epochs=8,                    # Long warmup
            warmup_latent_steps=45,
            warmup_latent_lr=0.009,

            precision_rescale="unit_mean_precision",
            psi0=300.0,                         # Stronger prior
            nu0_extra=6.0,
        ),
        "mnist_final_push": replace(
            base,
            name="mnist_final_push",
            hidden_layers=2,
            normalize="zero_one",

            hidden_init="feedforward",
            latent_steps=40,              # More inference time
            latent_lr=0.002,

            update_mode="svi_power",
            scale_power=0.15,             # Even gentler
            kappa_delay=4000.0,
            kappa_max=0.045,

            warmup_epochs=10,             # Longer warmup
            warmup_latent_steps=50,
            warmup_latent_lr=0.008,

            precision_rescale="unit_mean_precision",
            psi0=350.0,                   # Slightly stronger prior
            nu0_extra=8.0,
        ),
        "paper_exact_mnist": replace(
            base,
            name="paper_exact_mnist",
            hidden_layers=3,          # 3 hidden + 1 output = 4-layer network
            hidden=128,
            normalize="zero_one",     # pixels in [0,1]
            update_mode="batch_local",# no SVI rescaling — paper doesn't mention it
            kappa_delay=0.0,          # no delay in paper
            kappa_max=None,           # no cap in paper
            kappa_exponent=0.25,      # paper: epsilon = 0.25
            hidden_init="feedforward",# warm start — essential for alive gradients
            latent_steps=10,          # paper: 10 iterations
            latent_lr=0.01,           # paper: Adam lr=0.01
        ),
        "paper_stabilized_mnist": replace(
            base,
            name="paper_stabilized_mnist",
            hidden_layers=3,
            hidden=128,
            normalize="zero_one",
            update_mode="batch_local",
            kappa_delay=0.0,
            kappa_max=0.5,            # gentle cap to prevent huge early jumps
            kappa_exponent=0.25,
            hidden_init="feedforward",
            latent_steps=20,          # more steps for convergence
            latent_lr=0.005,          # lower lr with more steps
        ),
        "mnist_v6": replace(
            base,
            name="mnist_v6",
            hidden_layers=2,
            hidden=128,
            normalize="zero_one",
            update_mode="online_additive",        # no prior-anchor forgetting
            kappa_exponent=0.25,
            kappa_delay=0.0,
            kappa_max=0.5,
            kappa_min=0.01,
            hidden_init="feedforward",
            latent_steps=20,
            latent_lr=0.005,
            psi0=1000.0,
            nu0_extra=2.0,
            precision_rescale="unit_mean_precision",
            warmup_epochs=3,
            warmup_latent_steps=30,
            warmup_latent_lr=0.008,
            epochs=50,
        ),
    }
