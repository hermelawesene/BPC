"""Small tensor operations shared by the BPC implementation."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg

from bpc.config import Array, DTYPE, JITTER, MIN_EIG


def sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def eye(n: int, dtype=DTYPE) -> Array:
    return jnp.eye(n, dtype=dtype)


def inv_spd(A: Array, jitter: float = JITTER) -> Array:
    A = sym(A) + jnp.asarray(jitter, dtype=A.dtype) * eye(A.shape[0], A.dtype)
    L = jnp.linalg.cholesky(A)
    I = eye(A.shape[0], A.dtype)
    return jsp_linalg.cho_solve((L, True), I)


def project_spd(A: Array, min_eig: float = MIN_EIG) -> Array:
    A = sym(A)
    w, Q = jnp.linalg.eigh(A)
    w = jnp.maximum(w, jnp.asarray(min_eig, dtype=A.dtype))
    return sym((Q * w) @ Q.T)


def augment(z: Array) -> Array:
    return jnp.concatenate([z, jnp.ones((z.shape[0], 1), dtype=z.dtype)], axis=-1)


_sym = sym
_eye = eye
_inv_spd = inv_spd
_project_spd = project_spd
