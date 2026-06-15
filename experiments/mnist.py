from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np

from bpc.config import SELECTED_PRESET, make_presets
from bpc.data import build_mnist_dims, load_mnist
from bpc.training.trainer import train_mnist_dataset
from bpc.training.calibration import evaluate_calibration, plot_reliability_diagram
from bpc.utils.config_io import load_experiment_config
from experiments.evaluate_calibration import save_params, print_comparison_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the faithful BPC MNIST experiment.")
    parser.add_argument("--config",     default="configs/mnist.yaml",
                        help="Path to the YAML experiment config.")
    parser.add_argument("--preset",     default=None,
                        help="Optional preset override.")
    parser.add_argument("--mc_samples", type=int, default=50,
                        help="MC samples for calibration evaluation (default: 50).")
    parser.add_argument("--skip_calib", action="store_true",
                        help="Skip calibration evaluation after training.")
    args = parser.parse_args()

    default_cfg = make_presets()[SELECTED_PRESET]
    cfg, lcfg, raw = load_experiment_config(args.config, default_cfg, args.preset)

    if raw.get("mode") == "sweep":
        summary = []
        for name in raw.get("sweep_presets", []):
            sweep_cfg = make_presets()[name]
            data = load_mnist(sweep_cfg)
            layer_dims = build_mnist_dims(sweep_cfg)
            _, metrics, _ = train_mnist_dataset(sweep_cfg, data, layer_dims, lcfg)
            summary.append((name, metrics["best_acc"], metrics["best_epoch"], metrics["final_acc"]))
        print(json.dumps({"summary": summary}, indent=2, default=str))
        return

    data = load_mnist(cfg)
    layer_dims = build_mnist_dims(cfg)
    best_params, metrics, _ = train_mnist_dataset(cfg, data, layer_dims, lcfg)
    print(json.dumps(metrics, indent=2, default=str))

    if args.skip_calib:
        return

    preset_name = args.preset or cfg.name
    out_dir = Path("results") / "calibration" / preset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_params(best_params, str(out_dir / "best_params.npz"))

    print("\n── Post-training calibration evaluation ────────────────────")
    x_train, y_train, x_test, y_test = data
    x_test_np = np.asarray(x_test)
    y_test_np = np.asarray(y_test)

    calib_metrics = evaluate_calibration(
        params      = best_params,
        layer_dims  = layer_dims,
        x_test      = x_test_np,
        y_test_oh   = y_test_np,
        cfg         = cfg,
        mc_samples  = args.mc_samples,
        batch_size  = 512,
        seed        = 0,
        verbose     = True,
    )

    json_path = out_dir / "calibration_metrics.json"
    with open(json_path, "w") as f:
        json.dump(calib_metrics, f, indent=2)
    print(f"  Calibration metrics saved {json_path}")

    print_comparison_table(calib_metrics, preset_name)

    plot_reliability_diagram(
        params      = best_params,
        layer_dims  = layer_dims,
        x_test      = x_test_np,
        y_test_oh   = y_test_np,
        cfg         = cfg,
        save_path   = str(out_dir / "reliability_diagram.png"),
        mc_samples  = args.mc_samples,
        batch_size  = 512,
        seed        = 0,
    )


if __name__ == "__main__":
    main()