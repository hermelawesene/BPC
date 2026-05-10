"""Convergence helpers for inspecting hidden-state inference traces."""

from __future__ import annotations

import jax.numpy as jnp

from bpc.config import Array
from bpc.posterior.posterior_state import LatentDiagnostics


def final_gradient_max(diag: LatentDiagnostics) -> Array:
    """Return the maximum final hidden-state gradient norm."""

    return jnp.max(diag.final_grad_norm)
