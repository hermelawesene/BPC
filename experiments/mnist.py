"""Thin MNIST experiment entry point."""

from __future__ import annotations

import argparse
import json

from bpc.config import SELECTED_PRESET, make_presets
from bpc.data import build_mnist_dims, load_mnist
from bpc.training.trainer import train_mnist_dataset
from bpc.utils.config_io import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the faithful BPC MNIST experiment.")
    parser.add_argument("--config", default="configs/mnist.yaml", help="Path to the YAML experiment config.")
    parser.add_argument("--preset", default=None, help="Optional preset override.")
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
    _, metrics, _ = train_mnist_dataset(cfg, data, layer_dims, lcfg)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
