"""Adam-based hidden-state inference."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from bpc.config import BPCConfig, Array, DTYPE, ENABLE_X64
from bpc.inference.gradients import bpc_hidden_gradients, bpc_hidden_gradients_with_energy
from bpc.posterior.posterior_state import LatentDiagnostics, LayerMoments


def adam_latent_inference(
    hidden0: Tuple[Array, ...],
    moms: Tuple[LayerMoments, ...],
    x: Array,
    y: Array,
    cfg: BPCConfig,
) -> Tuple[Array, ...]:
    m = jax.tree.map(jnp.zeros_like, hidden0)
    v = jax.tree.map(jnp.zeros_like, hidden0)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-12 if ENABLE_X64 else 1e-8

    def body(carry, i):
        hidden, m, v = carry
        g = bpc_hidden_gradients(hidden, moms, x, y, cfg)
        m = jax.tree.map(lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi, m, g)
        v = jax.tree.map(lambda vi, gi: beta2 * vi + (1.0 - beta2) * gi * gi, v, g)
        t = i + 1
        m_hat = jax.tree.map(lambda mi: mi / (1.0 - beta1 ** t), m)
        v_hat = jax.tree.map(lambda vi: vi / (1.0 - beta2 ** t), v)
        hidden = jax.tree.map(lambda zi, mi, vi: zi - cfg.latent_lr * mi / (jnp.sqrt(vi) + eps), hidden, m_hat, v_hat)
        return (hidden, m, v), None

    (hidden, _, _), _ = jax.lax.scan(body, (hidden0, m, v), jnp.arange(cfg.latent_steps))
    return hidden


def adam_latent_inference_diag(
    hidden0: Tuple[Array, ...],
    moms: Tuple[LayerMoments, ...],
    x: Array,
    y: Array,
    cfg: BPCConfig,
) -> Tuple[Tuple[Array, ...], LatentDiagnostics]:
    T = int(cfg.latent_steps)
    L_state = len(hidden0)
    L_eta = len(moms)

    grad_log = jnp.zeros((T, L_state), dtype=DTYPE)
    z_log = jnp.zeros((T, L_state), dtype=DTYPE)
    e_log = jnp.zeros((T, L_eta), dtype=DTYPE)

    m = jax.tree.map(jnp.zeros_like, hidden0)
    v = jax.tree.map(jnp.zeros_like, hidden0)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-12 if ENABLE_X64 else 1e-8

    def body(carry, i):
        hidden, m, v, gl, zl, el = carry
        g, energies = bpc_hidden_gradients_with_energy(hidden, moms, x, y, cfg)
        gnorm = jnp.stack([jnp.linalg.norm(gi) for gi in g])
        znorm = jnp.stack([jnp.linalg.norm(hi) for hi in hidden])
        gl = gl.at[i].set(gnorm)
        zl = zl.at[i].set(znorm)
        el = el.at[i].set(energies)
        m = jax.tree.map(lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi, m, g)
        v = jax.tree.map(lambda vi, gi: beta2 * vi + (1.0 - beta2) * gi * gi, v, g)
        t = i + 1
        m_hat = jax.tree.map(lambda mi: mi / (1.0 - beta1 ** t), m)
        v_hat = jax.tree.map(lambda vi: vi / (1.0 - beta2 ** t), v)
        hidden = jax.tree.map(lambda zi, mi, vi: zi - cfg.latent_lr * mi / (jnp.sqrt(vi) + eps), hidden, m_hat, v_hat)
        return (hidden, m, v, gl, zl, el), None

    (hidden, _, _, gl, zl, el), _ = jax.lax.scan(
        body, (hidden0, m, v, grad_log, z_log, e_log), jnp.arange(T)
    )

    M_norm = jnp.stack([jnp.linalg.norm(mom.M) for mom in moms])
    Vinv_fro = jnp.stack([jnp.linalg.norm(mom.V) for mom in moms])
    psi_inv_raw_min = jnp.stack([mom.psi_inv_raw_min for mom in moms])
    E_prec_diag_mean = jnp.stack([jnp.mean(jnp.diag(mom.E_prec)) for mom in moms])

    diag = LatentDiagnostics(
        grad_norm_per_step=gl,
        latent_norm_per_step=zl,
        energy_per_step=el,
        init_grad_norm=gl[0],
        final_grad_norm=gl[-1],
        M_norm=M_norm,
        Vinv_fro=Vinv_fro,
        psi_inv_raw_min=psi_inv_raw_min,
        E_prec_diag_mean=E_prec_diag_mean,
    )
    return hidden, diag
