"""Dataset loading and batching helpers for BPC experiments."""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from bpc.config import BPCConfig, DTYPE, ENABLE_X64


def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float64 if ENABLE_X64 else np.float32)[labels.astype(np.int64)]


_one_hot = one_hot


def generate_two_moons(n_train: int = 1000, n_test: int = 300, noise: float = 0.1, seed: int = 0):
    try:
        from sklearn.datasets import make_moons
        x_train, y_train = make_moons(n_samples=n_train, noise=noise, random_state=seed)
        x_test, y_test = make_moons(n_samples=n_test, noise=noise, random_state=seed + 1)
    except Exception:
        def custom(n: int, off: int):
            r = np.random.default_rng(seed + off)
            m = n // 2
            th = np.linspace(0, np.pi, m)
            x1 = np.column_stack([np.cos(th), np.sin(th)])
            x2 = np.column_stack([1 - np.cos(th), 0.5 - np.sin(th)])
            X = np.vstack([x1, x2])
            X += r.normal(0.0, noise, X.shape)
            y = np.array([0] * m + [1] * m)
            p = r.permutation(n)
            return X[p], y[p]
        x_train, y_train = custom(n_train, 0)
        x_test, y_test = custom(n_test, 1)
    return (
        jnp.asarray(x_train, dtype=DTYPE),
        jnp.asarray(np.eye(2)[y_train.astype(int)], dtype=DTYPE),
        jnp.asarray(x_test, dtype=DTYPE),
        jnp.asarray(np.eye(2)[y_test.astype(int)], dtype=DTYPE),
    )


_MNIST_CACHE: Dict[Tuple[str, Optional[str], str], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def load_mnist(cfg: BPCConfig):
    key = (cfg.mnist_source, cfg.mnist_npz, cfg.normalize)
    if key in _MNIST_CACHE:
        return _MNIST_CACHE[key]

    if cfg.mnist_source == "npz":
        if cfg.mnist_npz is None:
            raise ValueError("mnist_npz must be set when mnist_source='npz'.")
        data = np.load(cfg.mnist_npz)
        x_train = data["x_train"].reshape((-1, 784)).astype(np.float64 if ENABLE_X64 else np.float32)
        x_test = data["x_test"].reshape((-1, 784)).astype(np.float64 if ENABLE_X64 else np.float32)
        y_train = data["y_train"]
        y_test = data["y_test"]
    else:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((-1, 784)).astype(np.float64 if ENABLE_X64 else np.float32)
        x_test = x_test.reshape((-1, 784)).astype(np.float64 if ENABLE_X64 else np.float32)

    if x_train.max() > 1.5:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    if cfg.normalize == "zero_one":
        pass
    elif cfg.normalize == "standardize_pixel":
        mu = x_train.mean(axis=0, keepdims=True)
        sd = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mu) / sd
        x_test = (x_test - mu) / sd
    elif cfg.normalize == "standardize_scalar":
        mu = float(x_train.mean())
        sd = float(x_train.std() + 1e-6)
        x_train = (x_train - mu) / sd
        x_test = (x_test - mu) / sd
    elif cfg.normalize == "centered_m11":
        x_train = 2.0 * x_train - 1.0
        x_test = 2.0 * x_test - 1.0
    else:
        raise ValueError(f"Unknown normalize={cfg.normalize}")

    out = (x_train, one_hot(y_train, 10), x_test, one_hot(y_test, 10))
    _MNIST_CACHE[key] = out
    return out


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = x.shape[0]
    idxs = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        ids = idxs[start:start + batch_size]
        yield x[ids], y[ids]


def compute_data_stats(x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "x_shape": list(x_train.shape),
        "x_min": float(x_train.min()),
        "x_max": float(x_train.max()),
        "x_mean": float(x_train.mean()),
        "x_std": float(x_train.std()),
        "x_dtype": str(x_train.dtype),
    }
    if y_train.ndim > 1:
        labels = y_train.argmax(axis=-1)
    else:
        labels = y_train
    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).tolist()
    stats["class_balance"] = counts
    stats["n_train"] = int(x_train.shape[0])
    return stats


def build_mnist_dims(cfg: BPCConfig) -> Tuple[int, ...]:
    return (784,) + tuple([cfg.hidden] * cfg.hidden_layers) + (10,)