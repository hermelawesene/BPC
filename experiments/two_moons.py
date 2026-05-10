"""Thin two-moons experiment entry point."""

from __future__ import annotations

import argparse

from sys import path
path.append(".")

from bpc.config import TWO_MOONS_NOISE
from bpc.data import generate_two_moons
from bpc.training.trainer import default_two_moons_config, train_two_moons_dataset
from bpc.utils.config_io import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the faithful BPC two-moons experiment.")
    parser.add_argument("--config", default="configs/two_moons.yaml", help="Path to the YAML experiment config.")
    parser.add_argument("--preset", default=None, help="Optional preset override.")
    args = parser.parse_args()

    cfg, lcfg, raw = load_experiment_config(args.config, default_two_moons_config(), args.preset)
    data_cfg = raw.get("data", {})
    n_train = int(data_cfg.get("n_train", 1000))
    n_test = int(data_cfg.get("n_test", 300))
    noise = float(data_cfg.get("noise", TWO_MOONS_NOISE))
    layer_dims = (2, cfg.hidden, 2)
    data = generate_two_moons(n_train=n_train, n_test=n_test, noise=noise, seed=cfg.seed)
    train_two_moons_dataset(cfg, data, layer_dims, lcfg)


if __name__ == "__main__":
    main()
