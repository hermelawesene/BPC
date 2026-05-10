"""High-level training loops for the faithful BPC implementation."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from bpc.config import (
    BPCConfig,
    DTYPE,
    LOGGING,
    REPORT_TRAIN_SUBSET_ACC,
    SAVE_DIR,
    SEED,
    STOP_ON_NAN,
    TRAIN_SUBSET_SIZE,
    TWO_MOONS_EPOCHS,
    TWO_MOONS_HIDDEN,
    TWO_MOONS_NOISE,
    LoggingConfig,
)
from bpc.data import batch_iterator, build_mnist_dims, compute_data_stats, generate_two_moons, load_mnist
from bpc.posterior.natural_params import Eta
from bpc.priors.weight_prior import init_posterior, init_prior
from bpc.training.metrics import classification_accuracy, has_bad_values, parameter_diagnostics, per_class_accuracy
from bpc.training.train_step import (
    make_apply_exact_or_ema_with_diag,
    make_apply_target_with_diag,
    make_infer_stats_with_diag,
    make_two_moons_epoch_with_diag,
)
from bpc.updates.hebbian_updates import add_stats, zero_stats
from bpc.updates.minibatch_updates import compute_kappa, compute_scale
from bpc.utils.logging import RunLogger
from bpc.utils.validation import validate_config, validate_layer_dims


MNDataset = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
JaxDataset = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def default_two_moons_config(seed: int = SEED) -> BPCConfig:
    return BPCConfig(name="two_moons", seed=seed, epochs=TWO_MOONS_EPOCHS, hidden=TWO_MOONS_HIDDEN, hidden_layers=1)


def train_two_moons_dataset(
    cfg: BPCConfig,
    data: JaxDataset,
    layer_dims: Tuple[int, ...],
    lcfg: Optional[LoggingConfig] = None,
):
    validate_config(cfg)
    validate_layer_dims(layer_dims)
    lcfg = lcfg if lcfg is not None else LOGGING
    print("\n" + "=" * 90)
    print("Two moons BPC - full-batch Eq. 7")
    print("=" * 90)

    x_train, y_train, x_test, y_test = data
    key = jax.random.PRNGKey(cfg.seed)
    prior = init_prior(layer_dims, v0=cfg.v0, psi0=cfg.psi0, nu0_extra=cfg.nu0_extra)
    params = init_posterior(layer_dims, key, v0=cfg.v0, psi0=cfg.psi0, nu0_extra=cfg.nu0_extra)

    logger = RunLogger(cfg, lcfg, layer_dims, mode="two_moons")
    try:
        data_stats = compute_data_stats(np.asarray(x_train), np.asarray(y_train))
    except Exception:
        data_stats = None
    logger.write_manifest(jax.devices(), prior, params, data_stats)

    one_epoch_with_diag = make_two_moons_epoch_with_diag(cfg, layer_dims, prior, x_train, y_train)

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        key, sub = jax.random.split(key)
        new_params, ldiag, sdiag, mdiag = one_epoch_with_diag(params, sub)
        params = new_params
        logger.log_batch(epoch, epoch, 0, 1.0, 1.0, sdiag, ldiag, mdiag)

        acc = classification_accuracy(params, layer_dims, x_test, y_test, cfg)
        row: Dict[str, object] = {
            "config": cfg.name,
            "epoch": epoch,
            "test_acc": acc,
            "time_sec": time.time() - ep_t0,
            "global_step": epoch,
        }
        if lcfg.parameter_diag_every_epoch:
            row.update(parameter_diagnostics(params, layer_dims))
        if lcfg.per_class_acc:
            try:
                pc = per_class_accuracy(params, layer_dims, x_test, y_test, cfg)
                for c, v in pc.items():
                    row[f"acc_class_{c}"] = v
            except Exception as e:
                print(f"per_class_accuracy skipped: {e}")
        rows.append(row)
        logger.log_epoch(row)

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"epoch {epoch:3d}/{cfg.epochs}: two_moons_acc={acc * 100:.2f}%")

    acc = classification_accuracy(params, layer_dims, x_test, y_test, cfg)
    print(f"final two_moons_acc={acc * 100:.2f}% time={time.time() - t0:.1f}s")
    try:
        logger.write_plots(np.asarray(x_train))
    except Exception as e:
        print(f"plots skipped: {e}")
    logger.close()
    return params, layer_dims


def train_two_moons(seed: int = SEED, lcfg: Optional[LoggingConfig] = None):
    cfg = default_two_moons_config(seed)
    layer_dims = (2, TWO_MOONS_HIDDEN, 2)
    data = generate_two_moons(noise=TWO_MOONS_NOISE, seed=seed)
    return train_two_moons_dataset(cfg, data, layer_dims, lcfg)


def train_mnist_dataset(
    cfg: BPCConfig,
    data: MNDataset,
    layer_dims: Tuple[int, ...],
    lcfg: Optional[LoggingConfig] = None,
) -> Tuple[Tuple[Eta, ...], Dict[str, float], List[Dict[str, object]]]:
    validate_config(cfg)
    validate_layer_dims(layer_dims)
    lcfg = lcfg if lcfg is not None else LOGGING
    save_dir = lcfg.save_dir or SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 100)
    print(f"MNIST BPC config: {cfg.name}")
    print("=" * 100)
    print("JAX devices:", jax.devices())
    print(json.dumps(asdict(cfg), indent=2))

    x_train_np, y_train_np, x_test_np, y_test_np = data
    n_total = int(x_train_np.shape[0])
    key = jax.random.PRNGKey(cfg.seed)
    prior = init_prior(layer_dims, v0=cfg.v0, psi0=cfg.psi0, nu0_extra=cfg.nu0_extra)
    params = init_posterior(layer_dims, key, v0=cfg.v0, psi0=cfg.psi0, nu0_extra=cfg.nu0_extra)
    best_params = params
    best_acc = -1.0
    best_epoch = 0
    rng = np.random.default_rng(cfg.seed)

    logger = RunLogger(cfg, lcfg, layer_dims, mode="mnist")
    data_stats = None
    if lcfg.log_data_stats:
        try:
            data_stats = compute_data_stats(x_train_np, y_train_np)
        except Exception as e:
            print(f"data stats skipped: {e}")
    logger.write_manifest(jax.devices(), prior, params, data_stats)

    x_test = jnp.asarray(x_test_np, dtype=DTYPE)
    y_test = jnp.asarray(y_test_np, dtype=DTYPE)
    x_train_eval = jnp.asarray(x_train_np[:TRAIN_SUBSET_SIZE], dtype=DTYPE) if REPORT_TRAIN_SUBSET_ACC else None
    y_train_eval = jnp.asarray(y_train_np[:TRAIN_SUBSET_SIZE], dtype=DTYPE) if REPORT_TRAIN_SUBSET_ACC else None

    warmup_cfg = replace(
        cfg,
        latent_steps=cfg.warmup_latent_steps if cfg.warmup_latent_steps is not None else cfg.latent_steps,
        latent_lr=cfg.warmup_latent_lr if cfg.warmup_latent_lr is not None else cfg.latent_lr,
        hidden_init=cfg.warmup_hidden_init if cfg.warmup_hidden_init is not None else cfg.hidden_init,
    )

    infer_stats_for_epoch = make_infer_stats_with_diag(cfg, warmup_cfg, layer_dims)
    apply_target_with_diag = make_apply_target_with_diag(cfg, layer_dims, prior)
    apply_exact_or_ema_with_diag = make_apply_exact_or_ema_with_diag(cfg, layer_dims, prior)

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    global_step = 1
    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        epoch_max_eta_delta = 0.0
        last_kappa = float("nan")
        last_scale = float("nan")
        batch_in_epoch = 0

        if cfg.update_mode in ("epoch_exact", "epoch_ema"):
            total_stats = zero_stats(layer_dims)
            buffered = []
            for xb_np, yb_np in batch_iterator(x_train_np, y_train_np, cfg.batch_size, rng, shuffle=True):
                xb = jnp.asarray(xb_np, dtype=DTYPE)
                yb = jnp.asarray(yb_np, dtype=DTYPE)
                key, sub = jax.random.split(key)
                st, ldiag, sdiag = infer_stats_for_epoch(params, xb, yb, sub, epoch)
                total_stats = add_stats(total_stats, st)
                buffered.append((global_step, batch_in_epoch, sdiag, ldiag))
                global_step += 1
                batch_in_epoch += 1
            kappa_v = compute_kappa(global_step, epoch, cfg)
            kappa = jnp.asarray(kappa_v, dtype=DTYPE)
            params, mdiag = apply_exact_or_ema_with_diag(params, total_stats, kappa)
            last_kappa = kappa_v
            last_scale = 1.0
            epoch_max_eta_delta = float(np.max(np.asarray(mdiag.eta_delta_Vinv)))
            for i, (gs, bi, sdiag_b, ldiag_b) in enumerate(buffered):
                last = i == len(buffered) - 1
                logger.log_batch(
                    gs, epoch, bi,
                    kappa_v if last else None,
                    1.0 if last else None,
                    sdiag_b, ldiag_b,
                    mdiag if last else None,
                )
        elif cfg.update_mode in ("batch_local", "svi_scaled", "svi_power", "online_additive"):
            for xb_np, yb_np in batch_iterator(x_train_np, y_train_np, cfg.batch_size, rng, shuffle=True):
                xb = jnp.asarray(xb_np, dtype=DTYPE)
                yb = jnp.asarray(yb_np, dtype=DTYPE)
                key, sub = jax.random.split(key)
                st, ldiag, sdiag = infer_stats_for_epoch(params, xb, yb, sub, epoch)
                kappa_v = compute_kappa(global_step, epoch, cfg)
                scale_v = compute_scale(n_total, xb_np.shape[0], cfg)
                kappa = jnp.asarray(kappa_v, dtype=DTYPE)
                scale = jnp.asarray(scale_v, dtype=DTYPE)
                params, mdiag = apply_target_with_diag(params, st, kappa, scale)
                logger.log_batch(global_step, epoch, batch_in_epoch, kappa_v, scale_v, sdiag, ldiag, mdiag)
                cur_max_delta = float(np.max(np.asarray(mdiag.eta_delta_Vinv)))
                if cur_max_delta > epoch_max_eta_delta:
                    epoch_max_eta_delta = cur_max_delta
                last_kappa = kappa_v
                last_scale = scale_v
                global_step += 1
                batch_in_epoch += 1
                if STOP_ON_NAN and has_bad_values(params):
                    print(f"Stopped early: NaN/Inf detected after update {global_step} in epoch {epoch}.")
                    logger.log_nan_event(global_step, epoch, params)
                    break
        else:
            raise ValueError(f"Unknown update_mode={cfg.update_mode}")

        test_acc = classification_accuracy(params, layer_dims, x_test, y_test, cfg)
        train_acc = float("nan")
        if REPORT_TRAIN_SUBSET_ACC and x_train_eval is not None and y_train_eval is not None:
            train_acc = classification_accuracy(params, layer_dims, x_train_eval, y_train_eval, cfg)

        if test_acc > best_acc and cfg.keep_best:
            best_acc = test_acc
            best_epoch = epoch
            best_params = params
        elif not cfg.keep_best:
            best_acc = test_acc
            best_epoch = epoch
            best_params = params

        row: Dict[str, object] = {
            "config": cfg.name,
            "epoch": epoch,
            "train10k_acc": train_acc,
            "test_acc": test_acc,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "time_sec": time.time() - ep_t0,
            "global_step": global_step,
            "max_eta_delta_Vinv_in_epoch": epoch_max_eta_delta,
            "last_kappa": last_kappa,
            "last_scale": last_scale,
        }
        if lcfg.parameter_diag_every_epoch:
            row.update(parameter_diagnostics(params, layer_dims))
        if lcfg.per_class_acc:
            try:
                pc = per_class_accuracy(params, layer_dims, x_test, y_test, cfg)
                for c, v in pc.items():
                    row[f"acc_class_{c}"] = v
            except Exception as e:
                print(f"per_class_accuracy skipped: {e}")
        rows.append(row)
        logger.log_epoch(row)
        print(
            f"epoch {epoch:3d}/{cfg.epochs}: train10k_acc={train_acc * 100:6.2f}% "
            f"test_acc={test_acc * 100:6.2f}% best={best_acc * 100:6.2f}%@{best_epoch} "
            f"time={row['time_sec']:.1f}s"
        )

        if STOP_ON_NAN and has_bad_values(params):
            break

    final_acc = rows[-1]["test_acc"] if rows else float("nan")
    diag = parameter_diagnostics(best_params if cfg.keep_best else params, layer_dims)
    print(f"final test_acc={final_acc * 100:.2f}%  best_acc={best_acc * 100:.2f}% at epoch {best_epoch}")
    print("diagnostics on selected params:", {k: round(v, 5) for k, v in list(diag.items())[:20]}, "...")
    print(f"total_time={time.time() - t0:.1f}s")

    csv_path = os.path.join(save_dir, "bpc_mnist_experiment_log.csv")
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as f:
            if rows:
                fieldnames = list(rows[0].keys()) + list(asdict(cfg).keys())
            else:
                fieldnames = list(asdict(cfg).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for r in rows:
                full = dict(r)
                full.update(asdict(cfg))
                writer.writerow(full)
        print("Legacy CSV log:", csv_path)
    except Exception as e:
        print(f"Legacy CSV skipped: {e}")

    try:
        logger.write_plots(x_train_np)
    except Exception as e:
        print(f"plots skipped: {e}")
    logger.close()

    return best_params if cfg.keep_best else params, {"final_acc": final_acc, "best_acc": best_acc, "best_epoch": best_epoch, **diag}, rows


def train_mnist(cfg: BPCConfig, lcfg: Optional[LoggingConfig] = None) -> Tuple[Tuple[Eta, ...], Dict[str, float], List[Dict[str, object]]]:
    data = load_mnist(cfg)
    layer_dims = build_mnist_dims(cfg)
    return train_mnist_dataset(cfg, data, layer_dims, lcfg)
