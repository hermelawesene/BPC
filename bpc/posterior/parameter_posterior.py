"""Posterior moment compatibility exports."""

from bpc.distributions.matrix_normal_wishart import natural_to_moment, posterior_moments
from bpc.priors.weight_prior import init_posterior

__all__ = ["init_posterior", "natural_to_moment", "posterior_moments"]
