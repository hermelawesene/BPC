from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from bpc.config import BPCConfig, DTYPE, Array
from bpc.data import load_mnist, build_mnist_dims
from bpc.distributions.matrix_normal_wishart import natural_to_moment, posterior_moments
from bpc.layers.predictive_network import predict
from bpc.posterior.natural_params import Eta
from bpc.activations import activation
from bpc.utils.tensor_ops import augment, sym


def _map_probs_batch(
    params: Tuple[Eta, ...],
    layer_dims: Tuple[int, ...],
    x: Array,
    cfg: BPCConfig,
) -> Array:
    logits = predict(params, layer_dims, x, cfg)
    return jax.nn.softmax(logits, axis=-1)


def _sample_weights(
    M: Array,
    V: Array,
    Psi: Array,
    nu: float,
    rng_key: Array,
) -> Array:
    d_y, d_x = M.shape

    k1, k2, k3 = jax.random.split(rng_key, 3)
    psi_inv = jnp.linalg.cholesky(Psi + 1e-7 * jnp.eye(d_y, dtype=Psi.dtype))
    A = jnp.tril(jax.random.normal(k1, (d_y, d_y), dtype=Psi.dtype))
    nu_vec = jnp.array([nu - i for i in range(d_y)], dtype=Psi.dtype)
    chi2_samples = jnp.sqrt(jax.random.chisquare(k2, df=nu_vec))
    A = A.at[jnp.arange(d_y), jnp.arange(d_y)].set(chi2_samples)
    L_W = psi_inv @ A                                       
    WW = L_W @ L_W.T
    WW_chol = jnp.linalg.cholesky(WW + 1e-7 * jnp.eye(d_y, dtype=WW.dtype))
    Sigma_chol = jnp.linalg.solve(WW_chol, jnp.eye(d_y, dtype=WW.dtype)).T  

    V_chol = jnp.linalg.cholesky(V + 1e-7 * jnp.eye(d_x, dtype=V.dtype))
    Z = jax.random.normal(k3, (d_y, d_x), dtype=M.dtype)
    W = M + Sigma_chol @ Z @ V_chol.T                      
    return W


def _mc_probs_batch(
    params: Tuple[Eta, ...],
    layer_dims: Tuple[int, ...],
    x: Array,
    cfg: BPCConfig,
    rng_key: Array,
    n_samples: int = 50,
) -> Array:
    moments = []
    for l, eta in enumerate(params):
        d_y = int(layer_dims[l + 1])
        d_x = int(layer_dims[l]) + 1
        M, V, Psi, nu, _ = natural_to_moment(eta, d_y, d_x)
        moments.append((M, V, Psi, nu))

    prob_accum = jnp.zeros((x.shape[0], int(layer_dims[-1])), dtype=DTYPE)
    for s in range(n_samples):
        rng_key, sub = jax.random.split(rng_key)
        keys = jax.random.split(sub, len(moments))
        z = x
        for l, (M, V, Psi, nu) in enumerate(moments):
            W_s = _sample_weights(M, V, Psi, nu, keys[l])
            z = augment(activation(z, l, cfg.input_activation)) @ W_s.T
        prob_accum = prob_accum + jax.nn.softmax(z, axis=-1)

    return prob_accum / n_samples


def _accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def _nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    n = len(labels)
    log_p = np.log(probs[np.arange(n), labels] + eps)
    return float(-log_p.mean())


def _brier(probs: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> float:
    one_hot = np.eye(n_classes)[labels]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(labels))


def _mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)
    mce = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(accuracies[mask].mean() - confidences[mask].mean())
        if gap > mce:
            mce = gap
    return float(mce)


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    H = -(probs * np.log(probs + eps)).sum(axis=1)
    return float(H.mean())


def _overconfidence(probs: np.ndarray, labels: np.ndarray) -> float:
    confidences  = probs.max(axis=1)
    correct      = (probs.argmax(axis=1) == labels).astype(float)
    return float((confidences > correct).mean())


def _calibration_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean_conf = np.zeros(n_bins)
    mean_acc  = np.zeros(n_bins)
    counts    = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        mean_conf[i] = confidences[mask].mean()
        mean_acc[i]  = accuracies[mask].mean()
        counts[i]    = mask.sum()
    return centres, mean_conf, mean_acc, counts

def evaluate_calibration(
    params: Tuple[Eta, ...],
    layer_dims: Tuple[int, ...],
    x_test: np.ndarray,
    y_test_oh: np.ndarray,
    cfg: BPCConfig,
    mc_samples: int = 50,
    batch_size: int = 512,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    labels = y_test_oh.argmax(axis=1)          
    N      = x_test.shape[0]
    n_cls  = int(layer_dims[-1])

    map_fn  = jax.jit(lambda p, xb: _map_probs_batch(p, layer_dims, xb, cfg))

    if verbose:
        print(f"  Running MAP predictions over {N} samples (batch={batch_size})...")
    t0 = time.time()
    map_probs_list = []
    for start in range(0, N, batch_size):
        xb = jnp.asarray(x_test[start:start + batch_size], dtype=DTYPE)
        pb = np.asarray(map_fn(params, xb))
        map_probs_list.append(pb)
    map_probs = np.concatenate(map_probs_list, axis=0)  
    if verbose:
        print(f"  MAP done in {time.time()-t0:.1f}s")

    if verbose:
        print(f"  Running MC predictions ({mc_samples} samples, batch={batch_size})...")
    t0 = time.time()
    rng = jax.random.PRNGKey(seed)
    mc_probs_list = []
    for start in range(0, N, batch_size):
        xb = jnp.asarray(x_test[start:start + batch_size], dtype=DTYPE)
        rng, sub = jax.random.split(rng)
        pb = np.asarray(
            _mc_probs_batch(params, layer_dims, xb, cfg, sub, n_samples=mc_samples)
        )
        mc_probs_list.append(pb)
        if verbose and (start // batch_size) % 5 == 0:
            print(f"    batch {start // batch_size + 1}/{(N + batch_size - 1) // batch_size}", end="\r")
    mc_probs = np.concatenate(mc_probs_list, axis=0)    
    if verbose:
        print(f"\n  MC done in {time.time()-t0:.1f}s")

    metrics: Dict[str, float] = {}

    metrics["map_accuracy_%"]   = round(_accuracy(map_probs, labels) * 100, 3)
    metrics["mc_accuracy_%"]    = round(_accuracy(mc_probs, labels) * 100, 3)

    metrics["map_nll"]          = round(_nll(map_probs, labels), 5)
    metrics["mc_nll"]           = round(_nll(mc_probs, labels), 5)

    metrics["map_brier"]        = round(_brier(map_probs, labels, n_cls), 5)
    metrics["mc_brier"]         = round(_brier(mc_probs, labels, n_cls), 5)

    metrics["map_ece"]          = round(_ece(map_probs, labels), 5)
    metrics["mc_ece"]           = round(_ece(mc_probs, labels), 5)

    metrics["map_mce"]          = round(_mce(map_probs, labels), 5)
    metrics["mc_mce"]           = round(_mce(mc_probs, labels), 5)

    metrics["map_entropy"]      = round(_entropy(map_probs), 5)
    metrics["mc_entropy"]       = round(_entropy(mc_probs), 5)

    metrics["map_overconfidence"] = round(_overconfidence(map_probs, labels), 5)
    metrics["mc_overconfidence"]  = round(_overconfidence(mc_probs, labels), 5)

    metrics["map_mean_confidence"] = round(float(map_probs.max(axis=1).mean()), 5)
    metrics["mc_mean_confidence"]  = round(float(mc_probs.max(axis=1).mean()), 5)

    for c in range(n_cls):
        mask = labels == c
        if mask.sum() > 0:
            metrics[f"mc_entropy_class_{c}"] = round(float(_entropy(mc_probs[mask])), 5)

    if verbose:
        print("\n── Calibration Metrics ─────────────────────────────────────")
        for k, v in metrics.items():
            if not k.startswith("mc_entropy_class"):
                print(f"  {k:<30} {v}")
        print("────────────────────────────────────────────────────────────\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram (calibration plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    params: Tuple[Eta, ...],
    layer_dims: Tuple[int, ...],
    x_test: np.ndarray,
    y_test_oh: np.ndarray,
    cfg: BPCConfig,
    save_path: Optional[str] = None,
    mc_samples: int = 50,
    batch_size: int = 512,
    seed: int = 0,
) -> None:
    """
    Plot reliability diagrams (confidence vs accuracy) for MAP and MC,
    plus entropy histograms for correct vs incorrect predictions.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping reliability diagram.")
        return

    labels  = y_test_oh.argmax(axis=1)
    map_fn  = jax.jit(lambda p, xb: _map_probs_batch(p, layer_dims, xb, cfg))
    rng     = jax.random.PRNGKey(seed)
    N       = x_test.shape[0]

    map_probs_list, mc_probs_list = [], []
    for start in range(0, N, batch_size):
        xb  = jnp.asarray(x_test[start:start + batch_size], dtype=DTYPE)
        map_probs_list.append(np.asarray(map_fn(params, xb)))
        rng, sub = jax.random.split(rng)
        mc_probs_list.append(np.asarray(_mc_probs_batch(params, layer_dims, xb, cfg, sub, mc_samples)))

    map_probs = np.concatenate(map_probs_list)
    mc_probs  = np.concatenate(mc_probs_list)

    # MAP calibration bins
    _, map_conf, map_acc, map_cnt = _calibration_bins(map_probs, labels)
    _, mc_conf,  mc_acc,  mc_cnt  = _calibration_bins(mc_probs, labels)

    # Entropy split by correctness
    map_correct = map_probs.argmax(axis=1) == labels
    mc_correct  = mc_probs.argmax(axis=1) == labels
    map_ent = -(map_probs * np.log(map_probs + 1e-12)).sum(axis=1)
    mc_ent  = -(mc_probs  * np.log(mc_probs  + 1e-12)).sum(axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("BPC MNIST Calibration Analysis (mnist_best_hybrid_v5)", fontsize=13, fontweight="bold")

    BLUE  = "#2E75B6"
    ORG   = "#DD8452"
    RED   = "#C0392B"

    # ── MAP reliability diagram ───────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(map_conf, map_acc, width=0.05, alpha=0.7, color=BLUE, label="MAP accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"MAP Reliability Diagram\nECE={_ece(map_probs,labels):.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── MC reliability diagram ────────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(mc_conf, mc_acc, width=0.05, alpha=0.7, color=ORG, label="MC accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"MC Reliability Diagram ({mc_samples} samples)\nECE={_ece(mc_probs,labels):.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Confidence histogram ──────────────────────────────────────────
    ax = axes[0, 2]
    ax.hist(map_probs.max(axis=1), bins=30, alpha=0.6, color=BLUE, label="MAP", density=True)
    ax.hist(mc_probs.max(axis=1),  bins=30, alpha=0.6, color=ORG,  label="MC",  density=True)
    ax.set_xlabel("Max softmax confidence"); ax.set_ylabel("Density")
    ax.set_title("Predictive Confidence Distribution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── MAP entropy: correct vs incorrect ─────────────────────────────
    ax = axes[1, 0]
    ax.hist(map_ent[map_correct],  bins=30, alpha=0.6, color=BLUE, label="Correct",   density=True)
    ax.hist(map_ent[~map_correct], bins=30, alpha=0.6, color=RED,  label="Incorrect", density=True)
    ax.set_xlabel("Predictive entropy"); ax.set_ylabel("Density")
    ax.set_title(f"MAP Entropy (correct vs incorrect)\nMean H={_entropy(map_probs):.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── MC entropy: correct vs incorrect ──────────────────────────────
    ax = axes[1, 1]
    ax.hist(mc_ent[mc_correct],  bins=30, alpha=0.6, color=ORG, label="Correct",   density=True)
    ax.hist(mc_ent[~mc_correct], bins=30, alpha=0.6, color=RED, label="Incorrect", density=True)
    ax.set_xlabel("Predictive entropy"); ax.set_ylabel("Density")
    ax.set_title(f"MC Entropy (correct vs incorrect)\nMean H={_entropy(mc_probs):.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Bar chart: MAP vs MC metrics ──────────────────────────────────
    ax = axes[1, 2]
    keys   = ["accuracy_%", "nll", "brier", "ece", "entropy"]
    labels_ = ["Accuracy (%)", "NLL ↓", "Brier ↓", "ECE ↓", "Entropy"]
    map_v  = [
        _accuracy(map_probs, labels) * 100,
        _nll(map_probs, labels),
        _brier(map_probs, labels),
        _ece(map_probs, labels),
        _entropy(map_probs),
    ]
    mc_v   = [
        _accuracy(mc_probs, labels) * 100,
        _nll(mc_probs, labels),
        _brier(mc_probs, labels),
        _ece(mc_probs, labels),
        _entropy(mc_probs),
    ]
    # Normalise to [0,1] per metric for display
    mx = [max(a, b, 1e-9) for a, b in zip(map_v, mc_v)]
    norm_map = [v / m for v, m in zip(map_v, mx)]
    norm_mc  = [v / m for v, m in zip(mc_v,  mx)]
    x_pos = np.arange(len(keys))
    ax.bar(x_pos - 0.2, norm_map, 0.35, color=BLUE, alpha=0.8, label="MAP")
    ax.bar(x_pos + 0.2, norm_mc,  0.35, color=ORG,  alpha=0.8, label="MC")
    for i, (mv, mcv, mk, mck) in enumerate(zip(map_v, mc_v, map_v, mc_v)):
        ax.text(i - 0.2, norm_map[i] + 0.02, f"{mv:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, norm_mc[i]  + 0.02, f"{mcv:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels_, fontsize=8)
    ax.set_title("MAP vs MC Metric Comparison\n(normalised per metric)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Reliability diagram saved → {save_path}")
    else:
        plt.show()
    plt.close()