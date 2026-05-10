"""Faithful modular refactor of the BPC experimenter."""

from bpc.config import BPCConfig, LoggingConfig, make_presets
from bpc.distributions.matrix_normal_wishart import natural_to_moment, posterior_moments
from bpc.layers.predictive_network import predict
from bpc.posterior.natural_params import Eta
from bpc.posterior.posterior_state import LayerMoments, LatentDiagnostics, MStepDiagnostics, Stats, StatsDiagnostics
from bpc.priors.weight_prior import init_posterior, init_prior
from bpc.training.metrics import classification_accuracy, parameter_diagnostics, per_class_accuracy

__all__ = [
    "BPCConfig",
    "Eta",
    "LayerMoments",
    "LatentDiagnostics",
    "LoggingConfig",
    "MStepDiagnostics",
    "Stats",
    "StatsDiagnostics",
    "classification_accuracy",
    "init_posterior",
    "init_prior",
    "make_presets",
    "natural_to_moment",
    "parameter_diagnostics",
    "per_class_accuracy",
    "posterior_moments",
    "predict",
]
