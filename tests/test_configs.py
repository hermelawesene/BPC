from __future__ import annotations

from dataclasses import asdict

import pytest

pytest.importorskip("jax")


def test_mnist_yaml_matches_selected_preset():
    from bpc.config import SELECTED_PRESET, make_presets
    from bpc.utils.config_io import load_experiment_config

    cfg, lcfg, raw = load_experiment_config("configs/mnist.yaml", make_presets()[SELECTED_PRESET])
    assert asdict(cfg) == asdict(make_presets()[SELECTED_PRESET])
    assert raw["preset"] == SELECTED_PRESET
    assert lcfg.save_dir == "runs"


def test_two_moons_yaml_matches_default_config():
    from bpc.config import TWO_MOONS_EPOCHS, TWO_MOONS_HIDDEN
    from bpc.training.trainer import default_two_moons_config
    from bpc.utils.config_io import load_experiment_config

    cfg, lcfg, raw = load_experiment_config("configs/two_moons.yaml", default_two_moons_config())
    assert cfg.name == "two_moons"
    assert cfg.epochs == TWO_MOONS_EPOCHS
    assert cfg.hidden == TWO_MOONS_HIDDEN
    assert cfg.hidden_layers == 1
    assert raw["data"]["noise"] == 0.10
    assert lcfg.save_dir == "runs"
