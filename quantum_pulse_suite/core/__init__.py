"""
Core module for quantum pulse sequence suite.

This module exports the main classes for pulse sequences, filter functions,
and noise generation.
"""

from .pulse_sequence import (
    # Constants
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    IDENTITY,
    # Utility functions
    normalize_axis,
    axis_from_spherical,
    # Pulse elements
    PulseElement,
    InstantaneousPulse,
    ContinuousPulse,
    FreeEvolution,
    # Pulse sequences
    PulseSequence,
    InstantaneousPulseSequence,
    ContinuousPulseSequence,
    # Factory functions
    ramsey_sequence,
    spin_echo_sequence,
    cpmg_sequence,
    continuous_rabi_sequence,
)

from .filter_functions import (
    # Fourier integrals
    cj,
    sj,
    # Filter function classes
    FilterFunction,
    InstantaneousFilterFunction,
    ContinuousFilterFunction,
    # Noise PSD factory
    ColoredNoisePSD,
)

from .noise import (
    generate_time_series,
    noise_interpolation,
    NoiseGenerator,
)

__all__ = [
    # Constants
    'SIGMA_X',
    'SIGMA_Y',
    'SIGMA_Z',
    'IDENTITY',
    # Utility functions
    'normalize_axis',
    'axis_from_spherical',
    # Pulse elements
    'PulseElement',
    'InstantaneousPulse',
    'ContinuousPulse',
    'FreeEvolution',
    # Pulse sequences
    'PulseSequence',
    'InstantaneousPulseSequence',
    'ContinuousPulseSequence',
    # Factory functions
    'ramsey_sequence',
    'spin_echo_sequence',
    'cpmg_sequence',
    'continuous_rabi_sequence',
    # Fourier integrals
    'cj',
    'sj',
    # Filter function classes
    'FilterFunction',
    'InstantaneousFilterFunction',
    'ContinuousFilterFunction',
    # Noise PSD factory
    'ColoredNoisePSD',
    # Noise generation
    'generate_time_series',
    'noise_interpolation',
    'NoiseGenerator',
]
