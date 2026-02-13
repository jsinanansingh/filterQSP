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

from .global_phase_spectroscopy import (
    GlobalPhaseSpectroscopySequence,
    GPSFilterFunction,
    gps_filter_functions_comparison,
    plot_gps_filter_functions,
)

from .detuning_optimization import (
    compute_signal,
    compute_noise_variance,
    compute_signal_slope,
    frequency_estimation_variance,
    optimize_detuning,
    OptimizationResult,
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
    # Global phase spectroscopy
    'GlobalPhaseSpectroscopySequence',
    'GPSFilterFunction',
    'gps_filter_functions_comparison',
    'plot_gps_filter_functions',
    # Detuning optimization
    'compute_signal',
    'compute_noise_variance',
    'compute_signal_slope',
    'frequency_estimation_variance',
    'optimize_detuning',
    'OptimizationResult',
]
