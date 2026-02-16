"""
Core module for quantum pulse sequence suite.

This module exports the main classes for pulse sequences, filter functions,
noise generation, Hilbert spaces, and operator algebras.
"""

from .hilbert_space import (
    HilbertSpace,
    Subspace,
    LiouvilleSpace,
)

from .operators import (
    # Pauli matrices (also in pulse_sequence for backward compat)
    SIGMA_X as PAULI_X,
    SIGMA_Y as PAULI_Y,
    SIGMA_Z as PAULI_Z,
    IDENTITY_2,
    pauli_matrices,
    gell_mann_matrices,
    generalized_gell_mann_matrices,
    rotation_operator,
    subspace_pauli,
    x_operator,
    y_operator,
    z_operator,
)

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
    continuous_ramsey_sequence,
    continuous_cpmg_sequence,
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

from .fft_filter_function import (
    fft_filter_function,
    noise_susceptibility_from_matrix,
    bloch_components_from_matrix,
)

from .noise import (
    generate_time_series,
    noise_interpolation,
    NoiseGenerator,
)

from .three_level_filter import (
    fft_three_level_filter,
    analytic_three_level_filter,
    three_level_noise_variance,
    Ff_analytic,
)

from .spin_displacement import (
    spin_j_operators,
    spin_displacement_hamiltonian,
    spin_displacement_pulse,
    snap_gate,
    su2_scaling_factor,
)

from .qudit_pulse_sequence import (
    QuditPulseElement,
    SpinDisplacementElement,
    InstantSpinDisplacement,
    QuditFreeEvolution,
    QuditPulseSequence,
)

from .irrep_decomposition import (
    angular_momentum_irrep_projectors,
    irrep_resolved_filter_function,
    transition_resolved_filter_function,
    ggm_component_filter_functions,
    irrep_multiplicities,
)

from .multilevel import (
    # Multi-level pulse elements
    MultiLevelPulseElement,
    MultiLevelInstantPulse,
    MultiLevelFreeEvolution,
    MultiLevelContinuousPulse,
    # Multi-level sequence
    MultiLevelPulseSequence,
    # Filter function for subspaces
    SubspaceFilterFunction,
    # Factory functions
    multilevel_ramsey,
    multilevel_spin_echo,
    multilevel_cpmg,
    # Differential spectroscopy
    DifferentialSpectroscopySequence,
)

__all__ = [
    # Hilbert space
    'HilbertSpace',
    'Subspace',
    'LiouvilleSpace',
    # Operator algebra
    'PAULI_X',
    'PAULI_Y',
    'PAULI_Z',
    'IDENTITY_2',
    'pauli_matrices',
    'gell_mann_matrices',
    'generalized_gell_mann_matrices',
    'rotation_operator',
    'subspace_pauli',
    'x_operator',
    'y_operator',
    'z_operator',
    # Constants (backward compat)
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
    # Multi-level support
    'MultiLevelPulseElement',
    'MultiLevelInstantPulse',
    'MultiLevelFreeEvolution',
    'MultiLevelContinuousPulse',
    'MultiLevelPulseSequence',
    'SubspaceFilterFunction',
    'multilevel_ramsey',
    'multilevel_spin_echo',
    'multilevel_cpmg',
    'DifferentialSpectroscopySequence',
    # Spin-displacement pulses
    'spin_j_operators',
    'spin_displacement_hamiltonian',
    'spin_displacement_pulse',
    'snap_gate',
    'su2_scaling_factor',
    # Qudit pulse sequences
    'QuditPulseElement',
    'SpinDisplacementElement',
    'InstantSpinDisplacement',
    'QuditFreeEvolution',
    'QuditPulseSequence',
    # Irrep decomposition
    'angular_momentum_irrep_projectors',
    'irrep_resolved_filter_function',
    'transition_resolved_filter_function',
    'ggm_component_filter_functions',
    'irrep_multiplicities',
    # FFT filter function
    'fft_filter_function',
    'noise_susceptibility_from_matrix',
    'bloch_components_from_matrix',
    # Three-level clock filter functions
    'fft_three_level_filter',
    'analytic_three_level_filter',
    'three_level_noise_variance',
    'Ff_analytic',
]
