"""Natural-parameter containers for Matrix-Normal Wishart factors."""

from __future__ import annotations

from typing import NamedTuple

from bpc.config import Array


class Eta(NamedTuple):
    """Natural parameters of one Matrix-Normal Wishart factor."""

    V_inv: Array
    MV: Array
    S: Array
    nu_shift: Array
