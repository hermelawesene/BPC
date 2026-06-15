from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from bpc.config import DTYPE, make_presets
from bpc.data import build_mnist_dims, load_mnist
from bpc.posterior.natural_params import Eta
from bpc.training.calibration import evaluate_calibration, plot_reliability_diagram
from bpc.training.trainer import train_mnist_dataset
from bpc.utils.config_io import load_experiment_config


def save_params(params: Tuple[Eta, ...], path: str) -> None:
    flat: dict = {}
    for l, eta in enumerate(params):
        flat[f"layer{l}_V_inv"]    = np.asarray(eta.V_inv)
        flat[f"layer{l}_MV"]       = np.asarray(eta.MV)
        flat[f"layer{l}_S"]        = np.asarray(eta.S)
        flat[f"layer{l}_nu_shift"] = np.asarray(eta.nu_shift)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **flat)
    print(f"  Params saved: {path}")


def load_params(path: str) -> Tuple[Eta, ...]:
    data = np.load(path)
    params = []
    l = 0
    while f"layer{l}_V_inv" in data:
        eta = Eta(
            V_inv    = jnp.asarray(data[f"layer{l}_V_inv"],    dtype=DTYPE),
            MV       = jnp.asarray(data[f"layer{l}_MV"],       dtype=DTYPE),
            S        = jnp.asarray(data[f"layer{l}_S"],        dtype=DTYPE),
            nu_shift = jnp.asarray(data[f"layer{l}_nu_shift"], dtype=DTYPE),
        )
        params.append(eta)
        l += 1
    print(f"  Params loaded ← {path} ({l} layers)")
    return tuple(params)

BBB_METRICS = {
    "map_accuracy_%":  96.85,
    "mc_accuracy_%":   96.66,
    "map_nll":         0.09816,
    "mc_nll":          0.12204,
    "map_brier":       0.04638,
    "mc_brier":        0.05419,
    "map_ece":         0.00225,
    "mc_ece":          0.02903,
}


def print_comparison_table(bpc_metrics: dict, preset: str) -> None:
    metrics_to_show = [
        ("map_accuracy_%",  "MAP Accuracy (%)",  "higher"),
        ("mc_accuracy_%",   "MC  Accuracy (%)",  "higher"),
        ("map_nll",         "MAP NLL",            "lower"),
        ("mc_nll",          "MC  NLL",            "lower"),
        ("map_brier",       "MAP Brier",          "lower"),
        ("mc_brier",        "MC  Brier",          "lower"),
        ("map_ece",         "MAP ECE",            "lower"),
        ("mc_ece",          "MC  ECE",            "lower"),
        ("map_entropy",     "MAP Entropy",        "info"),
        ("mc_entropy",      "MC  Entropy",        "info"),
        ("map_overconfidence", "MAP Overconf.",   "lower"),
        ("mc_overconfidence",  "MC  Overconf.",   "lower"),
    ]

    print("\n" + "=" * 72)
    print(f"  BBB vs BPC ({preset}) — MNIST Calibration Comparison")
    print("=" * 72)
    print(f"  {'Metric':<26} {'BBB':>10}  {'BPC':>10}  {'Better':>8}")
    print("-" * 72)
    for key, label, direction in metrics_to_show:
        bpc_val = bpc_metrics.get(key, float("nan"))
        bbb_val = BBB_METRICS.get(key, float("nan"))
        if direction == "higher":
            better = "BBB" if bbb_val >= bpc_val else "BPC"
        elif direction == "lower":
            better = "BBB" if bbb_val <= bpc_val else "BPC"
        else:
            better = "—"
        bbb_str = f"{bbb_val:.5f}" if not np.isnan(bbb_val) else "N/A"
        bpc_str = f"{bpc_val:.5f}" if not np.isnan(bpc_val) else "N/A"
        print(f"  {label:<26} {bbb_str:>10}  {bpc_str:>10}  {better:>8}")
    print("=" * 72 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BPC MNIST calibration evaluator")
    p.add_argument("--config",     default="configs/mnist.yaml",
                   help="YAML config path")
    p.add_argument("--preset",     default=None,
                   help="Config preset name (overrides YAML preset field)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .npz params file; if given, skip training")
    p.add_argument("--mc_samples", type=int, default=50,
                   help="MC weight samples for MC metrics (default: 50)")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Batch size for forward passes (default: 512)")
    p.add_argument("--out_dir",    default="results/calibration",
                   help="Output directory for metrics JSON and plots")
    p.add_argument("--seed",       type=int, default=0,
                   help="PRNG seed for MC sampling")
    p.add_argument("--no_plot",    action="store_true",
                   help="Skip reliability diagram plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    default_cfg = make_presets().get(
        args.preset or "mnist_best_hybrid_v5",
        list(make_presets().values())[0],
    )
    cfg, lcfg, _ = load_experiment_config(args.config, default_cfg, args.preset)
    preset_name  = args.preset or cfg.name
    out_dir      = Path(args.out_dir) / preset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BPC Calibration Evaluator")
    print(f"  Preset : {preset_name}")
    print(f"  MC     : {args.mc_samples} samples")
    print(f"  Out    : {out_dir}")
    print(f"{'='*60}\n")

    print("Loading MNIST...")
    data         = load_mnist(cfg)
    layer_dims   = build_mnist_dims(cfg)
    x_train, y_train, x_test, y_test = data
    print(f"  Test set: {x_test.shape[0]} samples, {layer_dims} dims")

    params: Tuple[Eta, ...]
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        params = load_params(args.checkpoint)
    else:
        print("\nNo checkpoint provided - training from scratch...")
        print("(Use --checkpoint path/to/params.npz to skip training)\n")
        t0 = time.time()
        params, metrics_train, _ = train_mnist_dataset(cfg, data, layer_dims, lcfg)
        print(f"\nTraining complete in {time.time()-t0:.1f}s")
        print(f"  Best acc : {metrics_train['best_acc']*100:.2f}% @ epoch {metrics_train['best_epoch']}")
        print(f"  Final acc: {metrics_train['final_acc']*100:.2f}%")

        ckpt_path = str(out_dir / "best_params.npz")
        save_params(params, ckpt_path)

    print(f"\nEvaluating calibration metrics...")
    x_test_np  = np.asarray(x_test)
    y_test_np  = np.asarray(y_test)

    metrics = evaluate_calibration(
        params      = params,
        layer_dims  = layer_dims,
        x_test      = x_test_np,
        y_test_oh   = y_test_np,
        cfg         = cfg,
        mc_samples  = args.mc_samples,
        batch_size  = args.batch_size,
        seed        = args.seed,
        verbose     = True,
    )

    json_path = out_dir / "calibration_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {json_path}")

    print_comparison_table(metrics, preset_name)

    if not args.no_plot:
        plot_path = str(out_dir / "reliability_diagram.png")
        print("Generating reliability diagram...")
        plot_reliability_diagram(
            params      = params,
            layer_dims  = layer_dims,
            x_test      = x_test_np,
            y_test_oh   = y_test_np,
            cfg         = cfg,
            save_path   = plot_path,
            mc_samples  = args.mc_samples,
            batch_size  = args.batch_size,
            seed        = args.seed,
        )

if __name__ == "__main__":
    main()