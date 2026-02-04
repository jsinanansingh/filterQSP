"""
Analysis module for quantum pulse sequence suite.

This module exports classes for numerical evolution, trajectory analysis,
and Allan variance computation for atomic clock experiments.
"""

from .batch_numerical import (
    bloch_vector_from_operator,
    NumericalEvolution,
    TrajectoryAnalyzer,
)

from .allan_variance import (
    compute_sensitivity_function,
    dick_effect_coefficients,
    allan_variance_dick,
    allan_variance_continuous,
    allan_deviation,
    allan_variance_vs_tau,
    quantum_projection_noise_limit,
)

__all__ = [
    'bloch_vector_from_operator',
    'NumericalEvolution',
    'TrajectoryAnalyzer',
    # Allan variance
    'compute_sensitivity_function',
    'dick_effect_coefficients',
    'allan_variance_dick',
    'allan_variance_continuous',
    'allan_deviation',
    'allan_variance_vs_tau',
    'quantum_projection_noise_limit',
]
