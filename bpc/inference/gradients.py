"""Hidden-state gradients and energy diagnostics."""

from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp

from bpc.activations import activation, activation_grad
from bpc.config import BPCConfig, Array, GRAD_NORM_CLIP
from bpc.posterior.posterior_state import LayerMoments
from bpc.utils.tensor_ops import augment


def bpc_hidden_gradients(
    hidden: Tuple[Array, ...],
    moms: Tuple[LayerMoments, ...],
    x: Array,
    y: Array,
    cfg: BPCConfig,
) -> Tuple[Array, ...]:
    z_all = (x,) + tuple(hidden) + (y,)
    grads: List[Array] = []
    n_layers = len(moms)
    for state_index in range(1, n_layers):
        z_l = z_all[state_index]

        mom_l = moms[state_index - 1]
        f_prev = augment(activation(z_all[state_index - 1], state_index - 1, cfg.input_activation))
        direct = z_l @ mom_l.E_prec.T - f_prev @ mom_l.E_prec_W.T

        mom_next = moms[state_index]
        f_l_aug = augment(activation(z_l, state_index, cfg.input_activation))
        grad_f_aug = f_l_aug @ mom_next.E_Wt_prec_W - z_all[state_index + 1] @ mom_next.E_prec_W
        topdown = grad_f_aug[:, : z_l.shape[1]] * activation_grad(z_l, state_index, cfg.input_activation)

        g = direct + topdown
        if GRAD_NORM_CLIP is not None:
            norm = jnp.linalg.norm(g)
            scale = jnp.minimum(1.0, jnp.asarray(GRAD_NORM_CLIP, dtype=g.dtype) / (norm + 1e-12))
            g = g * scale
        grads.append(g)
    return tuple(grads)


def bpc_hidden_gradients_with_energy(
    hidden: Tuple[Array, ...],
    moms: Tuple[LayerMoments, ...],
    x: Array,
    y: Array,
    cfg: BPCConfig,
) -> Tuple[Tuple[Array, ...], Array]:
    z_all = (x,) + tuple(hidden) + (y,)
    n_layers = len(moms)

    energies: List[Array] = []
    for l in range(n_layers):
        mom = moms[l]
        f_in = augment(activation(z_all[l], l, cfg.input_activation))
        z_out = z_all[l + 1]
        quad_z = jnp.sum(z_out * (z_out @ mom.E_prec.T))
        cross = jnp.sum(z_out * (f_in @ mom.E_prec_W.T))
        quad_f = jnp.sum(f_in * (f_in @ mom.E_Wt_prec_W))
        B = jnp.asarray(z_out.shape[0], dtype=z_out.dtype)
        e_l = 0.5 * (quad_z - 2.0 * cross + quad_f) / jnp.maximum(B, jnp.asarray(1.0, dtype=z_out.dtype))
        energies.append(e_l)

    grads: List[Array] = []
    for state_index in range(1, n_layers):
        z_l = z_all[state_index]
        mom_l = moms[state_index - 1]
        f_prev = augment(activation(z_all[state_index - 1], state_index - 1, cfg.input_activation))
        direct = z_l @ mom_l.E_prec.T - f_prev @ mom_l.E_prec_W.T

        mom_next = moms[state_index]
        f_l_aug = augment(activation(z_l, state_index, cfg.input_activation))
        grad_f_aug = f_l_aug @ mom_next.E_Wt_prec_W - z_all[state_index + 1] @ mom_next.E_prec_W
        topdown = grad_f_aug[:, : z_l.shape[1]] * activation_grad(z_l, state_index, cfg.input_activation)

        g = direct + topdown
        if GRAD_NORM_CLIP is not None:
            norm = jnp.linalg.norm(g)
            scale = jnp.minimum(1.0, jnp.asarray(GRAD_NORM_CLIP, dtype=g.dtype) / (norm + 1e-12))
            g = g * scale
        grads.append(g)

    energies_array = jnp.stack(energies)
    return tuple(grads), energies_array
