"""
Global Phase Spectroscopy analysis module.

This module implements the global phase spectroscopy protocol where:
1. System starts in clock superposition (|g> + |m>)/sqrt(2)
2. Continuous Rabi oscillation on optical/probe transition |g> <-> |e> for m cycles
3. Measurement on the metastable (clock) transition

The key insight is that only the |g> component of the clock superposition
participates in the Rabi oscillation, so the accumulated phase on |g> relative
to |m> encodes information about the probe transition frequency.

Theory
------
For m complete Rabi cycles with period T_Rabi = 2π/Ω on the probe:
- The |g> state accumulates a geometric phase
- The differential phase between |g> and |m> is sensitive to detuning
- Filter function describes sensitivity to frequency noise at different timescales

The filter function for this protocol differs from standard Ramsey:
- Continuous driving suppresses low-frequency noise (like spin echo)
- Sensitivity peaks near harmonics of the Rabi frequency
- Number of cycles m determines the spectral resolution

References
----------
- Zaporski et al., "Global Phase Spectroscopy"
"""

from typing import Tuple, Optional, Callable, List
import numpy as np
from scipy.linalg import expm

from ..systems.base import QuantumSystem
from ..core.hilbert_space import Subspace
from ..core.multilevel import (
    MultiLevelPulseSequence,
    MultiLevelContinuousPulse,
    SubspaceFilterFunction,
)
from ..core.filter_functions import cj, sj


class GlobalPhaseSpectroscopySequence:
    """
    Global phase spectroscopy with continuous Rabi oscillations.

    Implements the protocol:
    1. Prepare clock superposition (|g> + |m>)/sqrt(2)
    2. Drive m complete Rabi cycles on probe transition
    3. Measure on clock transition

    Parameters
    ----------
    system : QuantumSystem
        Three-level clock system
    n_cycles : int
        Number of complete Rabi cycles (m)
    omega : float
        Rabi frequency on probe transition
    delta : float, optional
        Detuning from probe resonance (default: 0)

    Examples
    --------
    >>> system = ThreeLevelClock()
    >>> gps = GlobalPhaseSpectroscopySequence(system, n_cycles=10, omega=2*np.pi)
    >>> ff = gps.get_filter_function_calculator()
    >>> Fx, Fy, Fz = ff.filter_function(frequencies)
    """

    def __init__(self, system: QuantumSystem, n_cycles: int,
                 omega: float, delta: float = 0.0):
        self._system = system
        self._n_cycles = n_cycles
        self._omega = omega
        self._delta = delta

        # Validate system has probe and clock subspaces
        if not hasattr(system, 'probe') or not hasattr(system, 'clock'):
            raise ValueError("System must have 'probe' and 'clock' subspaces")

        self._probe = system.probe
        self._clock = system.clock

        # Compute duration for m complete cycles
        # One cycle: Ω*T = 2π, so T_cycle = 2π/Ω
        self._t_cycle = 2 * np.pi / omega
        self._total_time = n_cycles * self._t_cycle

        # Build the pulse sequence
        self._sequence = self._build_sequence()

    @property
    def system(self) -> QuantumSystem:
        return self._system

    @property
    def n_cycles(self) -> int:
        return self._n_cycles

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def cycle_period(self) -> float:
        return self._t_cycle

    def _build_sequence(self) -> MultiLevelPulseSequence:
        """Build the continuous Rabi drive sequence."""
        seq = MultiLevelPulseSequence(self._system, self._probe)

        # Single continuous pulse for all cycles
        # Drive along x-axis
        seq.add_continuous_pulse(
            omega=self._omega,
            axis=[1, 0, 0],
            delta=self._delta,
            tau=self._total_time
        )

        return seq

    def compute_polynomials(self):
        """Compute QSP polynomial representation."""
        return self._sequence.compute_polynomials()

    def get_filter_function_calculator(self) -> 'GPSFilterFunction':
        """
        Get filter function calculator for this GPS sequence.

        Returns
        -------
        GPSFilterFunction
            Calculator for probe-transition filter functions
        """
        self._sequence.compute_polynomials()
        return GPSFilterFunction(
            self._sequence._poly_list,
            self._probe,
            self._omega,
            self._n_cycles,
            self._delta
        )

    def evolve_state(self, initial_state: np.ndarray) -> np.ndarray:
        """Evolve a state through the GPS sequence."""
        return self._sequence.evolve_state(initial_state)

    def differential_phase(self, initial_clock_phase: float = 0.0) -> float:
        """
        Compute the differential phase accumulated.

        Parameters
        ----------
        initial_clock_phase : float
            Initial phase in clock superposition

        Returns
        -------
        float
            Differential phase shift between |g> and |m>
        """
        psi0 = self._system.prepare_clock_superposition(initial_clock_phase)
        psi_final = self.evolve_state(psi0)
        return self._system.clock_phase(psi_final) - initial_clock_phase

    def contrast(self, initial_clock_phase: float = 0.0) -> float:
        """
        Compute the final clock contrast.

        Parameters
        ----------
        initial_clock_phase : float
            Initial phase

        Returns
        -------
        float
            Contrast in [0, 1]
        """
        psi0 = self._system.prepare_clock_superposition(initial_clock_phase)
        psi_final = self.evolve_state(psi0)
        return self._system.contrast(psi_final)

    def measurement_sensitivity(self, measurement_type: str = 'z') -> np.ndarray:
        """
        Get the measurement operator for different clock measurements.

        Parameters
        ----------
        measurement_type : str
            'z' : sigma_z on clock (population difference)
            'x' : sigma_x on clock (real part of coherence)
            'y' : sigma_y on clock (imaginary part of coherence)
            'population_g' : |g><g| (ground state population)
            'population_m' : |m><m| (metastable state population)

        Returns
        -------
        np.ndarray
            Measurement operator
        """
        sx_c, sy_c, sz_c = self._system.clock_pauli_matrices()

        if measurement_type == 'z':
            return sz_c
        elif measurement_type == 'x':
            return sx_c
        elif measurement_type == 'y':
            return sy_c
        elif measurement_type == 'population_g':
            return self._system.projector(self._system.GROUND)
        elif measurement_type == 'population_m':
            return self._system.projector(self._system.METASTABLE)
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")


class GPSFilterFunction(SubspaceFilterFunction):
    """
    Filter function calculator for Global Phase Spectroscopy.

    This specializes the filter function calculation for continuous
    Rabi driving, taking into account the specific structure of
    m complete Rabi cycles.

    Parameters
    ----------
    poly_list : list
        Polynomial data from sequence
    subspace : Subspace
        Probe subspace
    omega : float
        Rabi frequency
    n_cycles : int
        Number of cycles
    delta : float
        Detuning
    """

    def __init__(self, poly_list, subspace: Subspace,
                 omega: float, n_cycles: int, delta: float = 0.0):
        super().__init__(poly_list, subspace, continuous=True)
        self._omega = omega
        self._n_cycles = n_cycles
        self._delta = delta
        self._total_time = n_cycles * 2 * np.pi / omega

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def n_cycles(self) -> int:
        return self._n_cycles

    def filter_function_for_measurement(self, frequencies: np.ndarray,
                                         measurement_type: str = 'z') -> np.ndarray:
        """
        Compute filter function for a specific clock measurement using Kubo formula.

        The filter function depends on which observable is measured
        on the clock transition after the GPS sequence.

        For the 3-level clock system:
        - Probe drive acts on |g> <-> |e> subspace
        - Measurement is on |g> <-> |m> (clock) subspace
        - The |g> phase from probe dynamics appears in clock coherence

        The Kubo formula gives: |F|^2 - (B·F)^2
        where B is the measurement Bloch vector in the interaction frame.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies
        measurement_type : str
            'z', 'x', 'y', 'population_g', 'population_m'

        Returns
        -------
        np.ndarray
            Filter function |F(omega)|^2 for this measurement
        """
        Fx, Fy, Fz = self.filter_function(frequencies)

        # Get final probe rotation from polynomials
        B = self._get_effective_measurement_vector(measurement_type)

        # Kubo formula: |F|^2 - (B·F)^2
        F_mag_sq = Fx**2 + Fy**2 + Fz**2
        B_dot_F = B[0]*Fx + B[1]*Fy + B[2]*Fz
        sensitivity = F_mag_sq - B_dot_F**2

        return sensitivity

    def _get_effective_measurement_vector(self, measurement_type: str) -> np.ndarray:
        """
        Get the effective measurement Bloch vector in the probe interaction frame.

        For 3-level GPS:
        - The probe drive creates a rotation R on the {|g>, |e>} subspace
        - The clock measurement (on {|g>, |m>}) sees |g> as rotated
        - For complete Rabi cycles, R ≈ I, so B ≈ measurement direction

        The effective B accounts for how the clock measurement projects
        onto the probe dynamics.

        Parameters
        ----------
        measurement_type : str
            Type of clock measurement

        Returns
        -------
        np.ndarray
            Effective Bloch vector B = (Bx, By, Bz)
        """
        # Get final (f, g) from probe sequence
        if self._poly_list:
            f, g, omega, nx, ny, nz, tau = self._poly_list[-1]
            # For complete Rabi cycles, f ≈ ±1, g ≈ 0
            # Compute effective rotation
            f_final = self._compute_final_f(f, g, omega, tau)
            g_final = self._compute_final_g(f, g, omega, tau)
        else:
            f_final, g_final = 1.0, 0.0

        # Compute SO(3) rotation from SU(2) parameters
        R = self._su2_to_so3(f_final, g_final)

        # Measurement Bloch vector in lab frame
        if measurement_type == 'z':
            B_lab = np.array([0, 0, 1])
        elif measurement_type == 'x':
            B_lab = np.array([1, 0, 0])
        elif measurement_type == 'y':
            B_lab = np.array([0, 1, 0])
        elif measurement_type in ['population_g', 'population_m']:
            B_lab = np.array([0, 0, 1])  # Population is like σ_z
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

        # Transform to interaction frame: B = R^T · B_lab
        B = R.T @ B_lab

        return B

    def _compute_final_f(self, f0, g0, omega, tau):
        """Compute final f value after continuous pulse."""
        # For continuous Rabi with effective Rabi frequency
        rabi = np.sqrt(omega**2 + self._delta**2)
        if rabi < 1e-12:
            return f0
        theta = rabi * tau
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        # f evolution for x-axis drive
        return cos_half * f0 - 1j * (self._delta / rabi) * sin_half * f0

    def _compute_final_g(self, f0, g0, omega, tau):
        """Compute final g value after continuous pulse."""
        rabi = np.sqrt(omega**2 + self._delta**2)
        if rabi < 1e-12:
            return g0
        theta = rabi * tau
        sin_half = np.sin(theta / 2)
        # g evolution for x-axis drive
        return (omega / rabi) * sin_half

    def _su2_to_so3(self, f, g):
        """
        Convert SU(2) parameters (f, g) to SO(3) rotation matrix.

        For U = [[f, ig], [ig*, f*]], computes R such that
        U† σ_j U = Σ_k R_jk σ_k
        """
        a, b = np.real(f), np.imag(f)
        c, d = np.real(g), np.imag(g)

        R = np.array([
            [a**2 + c**2 - b**2 - d**2, 2*(c*d - a*b), 2*(a*d + b*c)],
            [2*(a*b + c*d), a**2 + d**2 - b**2 - c**2, 2*(b*d - a*c)],
            [2*(b*c - a*d), 2*(a*c + b*d), a**2 + b**2 - c**2 - d**2]
        ])
        return R

    def characteristic_frequencies(self) -> np.ndarray:
        """
        Return the characteristic frequencies for this GPS sequence.

        For m cycles of Rabi frequency Omega:
        - Fundamental: Omega
        - Harmonics: n * Omega for n = 1, 2, ...
        - Sequence length determines linewidth: ~1/T_total

        Returns
        -------
        np.ndarray
            Array of characteristic frequencies
        """
        harmonics = np.arange(1, 2 * self._n_cycles + 1) * self._omega
        return harmonics


def gps_filter_functions_comparison(system: QuantumSystem,
                                     n_cycles_list: List[int],
                                     omega: float,
                                     frequencies: np.ndarray,
                                     measurement_types: List[str] = None
                                     ) -> dict:
    """
    Compare filter functions for different GPS configurations.

    Parameters
    ----------
    system : QuantumSystem
        Three-level clock system
    n_cycles_list : list of int
        Different numbers of Rabi cycles to compare
    omega : float
        Rabi frequency
    frequencies : np.ndarray
        Frequencies at which to evaluate
    measurement_types : list of str, optional
        Measurement types to include (default: ['z', 'x', 'y'])

    Returns
    -------
    dict
        Results with structure:
        {
            'frequencies': frequencies,
            'n_cycles': n_cycles_list,
            'filter_functions': {
                n_cycles: {
                    measurement_type: |F(omega)|^2
                }
            }
        }
    """
    if measurement_types is None:
        measurement_types = ['z', 'x', 'y']

    results = {
        'frequencies': frequencies,
        'n_cycles': n_cycles_list,
        'measurement_types': measurement_types,
        'filter_functions': {},
        'total_times': {},
    }

    for n_cycles in n_cycles_list:
        gps = GlobalPhaseSpectroscopySequence(system, n_cycles=n_cycles, omega=omega)
        ff = gps.get_filter_function_calculator()

        results['total_times'][n_cycles] = gps.total_time
        results['filter_functions'][n_cycles] = {}

        for meas_type in measurement_types:
            ff_values = ff.filter_function_for_measurement(frequencies, meas_type)
            results['filter_functions'][n_cycles][meas_type] = ff_values

    return results


def plot_gps_filter_functions(results: dict, ax=None, normalize: bool = True):
    """
    Plot GPS filter functions from comparison results.

    Parameters
    ----------
    results : dict
        Output from gps_filter_functions_comparison
    ax : matplotlib axis, optional
        Axis to plot on (creates new figure if None)
    normalize : bool
        Whether to normalize filter functions by total time

    Returns
    -------
    matplotlib figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, len(results['measurement_types']),
                                  figsize=(4*len(results['measurement_types']), 4))
        if len(results['measurement_types']) == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]

    frequencies = results['frequencies']
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['n_cycles'])))

    for i, meas_type in enumerate(results['measurement_types']):
        ax = axes[i] if i < len(axes) else axes[0]

        for j, n_cycles in enumerate(results['n_cycles']):
            ff_values = results['filter_functions'][n_cycles][meas_type]
            total_time = results['total_times'][n_cycles]

            if normalize:
                ff_plot = ff_values / total_time**2
                ylabel = r'$|F(\omega)|^2 / T^2$'
            else:
                ff_plot = ff_values
                ylabel = r'$|F(\omega)|^2$'

            ax.semilogy(frequencies, ff_plot + 1e-20,
                       color=colors[j],
                       label=f'm={n_cycles}')

        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Measurement: clock $\\sigma_{meas_type}$')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
