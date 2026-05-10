# Bayesian Predictive Coding

This repository contains a JAX implementation of the paper **Bayesian
Predictive Coding** (arXiv:2503.24016v1), organized as a reusable `bpc/`
package with thin experiment entry points for two moons and MNIST.

## Layout

- `bpc/` contains reusable BPC implementation code:
  - `config.py`: defaults, presets, logging config, JAX dtype/setup.
  - `distributions/`, `posterior/`, `priors/`: Matrix-Normal Wishart natural
    parameters, prior/posterior initialization, and moment conversions.
  - `inference/`: hidden-state gradients, energy diagnostics, and Adam latent
    inference.
  - `updates/`: sufficient statistics, kappa/scale schedules, and posterior
    updates.
  - `training/`: JIT step factories, trainers, evaluation, and metrics.
  - `data.py`: MNIST/two-moons loading and batching.
  - `utils/logging.py`: run manifests, CSV/JSONL diagnostics, anomaly logs, and
    optional plots.
- `experiments/` contains only the two thin entry points.
- `configs/` contains YAML configs for the entry points.
- `tests/` contains minimal behavior tests.

## Run Experiments

Install the runtime dependencies in your environment first: `jax`, `jaxlib`,
`numpy`, and experiment-specific packages such as `scikit-learn`, `tensorflow`,
`matplotlib`, `pyyaml`, and `pytest` as needed.

Run two moons:

```bash
python -m experiments.two_moons --config configs/two_moons.yaml
```

Run MNIST:

```bash
python -m experiments.mnist --config configs/mnist.yaml
```

The YAML files override logging output to `runs/`. Algorithm defaults and
presets are inherited from `bpc.config`.

## Test

```bash
pytest
```

The tests cover core math, inference, update schedules, and MNIST NPZ
preprocessing. If JAX or NumPy are not installed, the tests skip rather than
downloading dependencies.
