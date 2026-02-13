"""
Quantum Pulse Suite - A library for quantum signal processing pulse sequences.

This package provides tools for designing, simulating, and analyzing quantum
pulse sequences, with support for both instantaneous and continuous pulses.

Main Features
-------------
- Pulse sequence construction with InstantaneousPulseSequence and ContinuousPulseSequence
- Filter function calculation for noise analysis
- Noise generation with various spectral densities
- Numerical evolution and trajectory analysis

Quick Start
-----------
>>> from quantum_pulse_suite import ramsey_sequence, spin_echo_sequence
>>> import numpy as np

# Create a spin echo sequence
>>> seq = spin_echo_sequence(tau=1.0, delta=0.5)
>>> print(f"Total duration: {seq.total_duration()}")

# Compute the filter function
>>> ff = seq.get_filter_function_calculator()
>>> freqs = np.logspace(-1, 2, 100)
>>> Fx, Fy, Fz = ff.filter_function(freqs)

# Get noise susceptibility
>>> susceptibility = ff.noise_susceptibility(freqs)

Examples
--------
Building a custom instantaneous pulse sequence:

>>> from quantum_pulse_suite import InstantaneousPulseSequence
>>> seq = InstantaneousPulseSequence()
>>> seq.add_instant_pulse([1, 0, 0], np.pi/2)  # pi/2 x-rotation
>>> seq.add_free_evolution(1.0, delta=0.1)     # 1.0 time unit, 0.1 detuning
>>> seq.add_instant_pulse([1, 0, 0], np.pi)    # pi x-rotation
>>> seq.add_free_evolution(1.0, delta=0.1)
>>> seq.add_instant_pulse([1, 0, 0], np.pi/2)

Building a continuous pulse sequence:

>>> from quantum_pulse_suite import ContinuousPulseSequence
>>> seq = ContinuousPulseSequence()
>>> seq.add_continuous_pulse(omega=np.pi, axis=[1, 0, 0], delta=0.0, tau=1.0)

Using noise generation:

>>> from quantum_pulse_suite import NoiseGenerator
>>> noise_gen = NoiseGenerator(seed=42)
>>> noise, times = noise_gen.generate(n_points=1000, dt=0.01, noise_type=1)

Numerical simulation:

>>> from quantum_pulse_suite.analysis import NumericalEvolution, TrajectoryAnalyzer
>>> evo = NumericalEvolution(dt=0.001)
>>> params = np.array([[np.pi, 1, 0, 0, 0, 1.0]])  # Single pi pulse on x-axis
>>> U_traj, times = evo.evolve_sequence(params)
"""

__version__ = "0.1.0"

# Import main classes from core module
from .core import (
    # Hilbert space (new)
    HilbertSpace,
    Subspace,
    LiouvilleSpace,
    # Operator algebra (new)
    generalized_gell_mann_matrices,
    gell_mann_matrices,
    pauli_matrices,
    subspace_pauli,
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
    continuous_ramsey_sequence,
    continuous_cpmg_sequence,
    # Fourier integrals
    cj,
    sj,
    # Filter function classes
    FilterFunction,
    InstantaneousFilterFunction,
    ContinuousFilterFunction,
    # Noise PSD factory
    ColoredNoisePSD,
    # FFT filter function
    fft_filter_function,
    noise_susceptibility_from_matrix,
    bloch_components_from_matrix,
    # Noise generation
    generate_time_series,
    noise_interpolation,
    NoiseGenerator,
)

# Import quantum system classes
from .systems import (
    QuantumSystem,
    QubitSystem,
    ThreeLevelClock,
)

# Import analysis classes
from .analysis import (
    bloch_vector_from_operator,
    NumericalEvolution,
    TrajectoryAnalyzer,
)

__all__ = [
    # Version
    '__version__',
    # Hilbert space
    'HilbertSpace',
    'Subspace',
    'LiouvilleSpace',
    # Operator algebra
    'generalized_gell_mann_matrices',
    'gell_mann_matrices',
    'pauli_matrices',
    'subspace_pauli',
    # Quantum systems
    'QuantumSystem',
    'QubitSystem',
    'ThreeLevelClock',
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
    'continuous_ramsey_sequence',
    'continuous_cpmg_sequence',
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
    # FFT filter function
    'fft_filter_function',
    'noise_susceptibility_from_matrix',
    'bloch_components_from_matrix',
    # Analysis
    'bloch_vector_from_operator',
    'NumericalEvolution',
    'TrajectoryAnalyzer',
]
