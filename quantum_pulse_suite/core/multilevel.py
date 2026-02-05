"""
Multi-level pulse sequences and filter functions.

This module extends the pulse sequence and filter function framework to work
with arbitrary d-dimensional quantum systems, allowing QSP sequences to be
applied on 2-level subspaces of larger systems.

Key Classes
-----------
MultiLevelPulseSequence
    Pulse sequence that operates on a subspace of a larger system.

SubspaceFilterFunction
    Filter function calculator for noise sensitivity on subspaces.

Example
-------
>>> from quantum_pulse_suite.systems import ThreeLevelClock
>>> system = ThreeLevelClock()
>>>
>>> # Create sequence on probe transition
>>> seq = MultiLevelPulseSequence(system, system.probe)
>>> seq.add_instant_pulse([1, 0, 0], np.pi/2)
>>> seq.add_free_evolution(tau=1.0)
>>> seq.add_instant_pulse([1, 0, 0], np.pi/2)
>>>
>>> # Compute filter function for probe dephasing
>>> ff = seq.get_filter_function_calculator()
>>> Fx, Fy, Fz = ff.filter_function(frequencies)
"""

from typing import List, Tuple, Optional, Dict, Callable, Union
import numpy as np
from scipy.linalg import expm

from ..systems.base import QuantumSystem
from ..core.hilbert_space import Subspace
from .filter_functions import FilterFunction, cj, sj


# =============================================================================
# Multi-level pulse elements
# =============================================================================

class MultiLevelPulseElement:
    """Base class for pulse elements in multi-level systems."""

    def __init__(self, subspace: Subspace):
        self._subspace = subspace
        self._dim = subspace.parent.dim

    @property
    def subspace(self) -> Subspace:
        return self._subspace

    @property
    def dim(self) -> int:
        return self._dim

    def duration(self) -> float:
        raise NotImplementedError

    def hamiltonian(self) -> np.ndarray:
        """Return full-space Hamiltonian."""
        raise NotImplementedError

    def unitary(self) -> np.ndarray:
        """Return full-space unitary."""
        raise NotImplementedError

    def subspace_unitary(self) -> np.ndarray:
        """Return 2x2 unitary projected to subspace."""
        U_full = self.unitary()
        return self._subspace.project_operator(U_full)


class MultiLevelInstantPulse(MultiLevelPulseElement):
    """
    Instantaneous rotation on a 2-level subspace.

    Parameters
    ----------
    subspace : Subspace
        The 2-level subspace to rotate
    axis : array-like
        Rotation axis [x, y, z] in subspace Bloch sphere
    angle : float
        Rotation angle in radians
    """

    def __init__(self, subspace: Subspace, axis: List[float], angle: float):
        super().__init__(subspace)
        axis = np.asarray(axis, dtype=float)
        self._axis = axis / np.linalg.norm(axis)
        self._angle = angle

    @property
    def axis(self) -> np.ndarray:
        return self._axis.copy()

    @property
    def angle(self) -> float:
        return self._angle

    def duration(self) -> float:
        return 0.0

    def hamiltonian(self) -> np.ndarray:
        """Return effective Hamiltonian (for reference)."""
        sx, sy, sz = self._subspace.subspace_pauli_matrices()
        n = self._axis
        return (n[0] * sx + n[1] * sy + n[2] * sz) / 2

    def unitary(self) -> np.ndarray:
        """Return rotation unitary in full space."""
        H = self.hamiltonian()
        return expm(-1j * H * self._angle)


class MultiLevelFreeEvolution(MultiLevelPulseElement):
    """
    Free evolution under detuning on a subspace.

    During free evolution, the system accumulates phase on the
    specified subspace transition.

    Parameters
    ----------
    subspace : Subspace
        The 2-level subspace for phase accumulation
    tau : float
        Duration of free evolution
    delta : float
        Detuning (frequency offset) on this transition
    """

    def __init__(self, subspace: Subspace, tau: float, delta: float = 0.0):
        super().__init__(subspace)
        self._tau = tau
        self._delta = delta

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def delta(self) -> float:
        return self._delta

    def duration(self) -> float:
        return self._tau

    def hamiltonian(self) -> np.ndarray:
        """Return free evolution Hamiltonian."""
        _, _, sz = self._subspace.subspace_pauli_matrices()
        return self._delta * sz / 2

    def unitary(self) -> np.ndarray:
        """Return free evolution unitary."""
        return expm(-1j * self.hamiltonian() * self._tau)


class MultiLevelContinuousPulse(MultiLevelPulseElement):
    """
    Continuous (finite-duration) pulse on a subspace.

    Parameters
    ----------
    subspace : Subspace
        The 2-level subspace to drive
    omega : float
        Rabi frequency
    axis : array-like
        Drive axis [x, y, z]
    delta : float
        Detuning
    tau : float
        Pulse duration
    """

    def __init__(self, subspace: Subspace, omega: float, axis: List[float],
                 delta: float, tau: float):
        super().__init__(subspace)
        axis = np.asarray(axis, dtype=float)
        self._axis = axis / np.linalg.norm(axis)
        self._omega = omega
        self._delta = delta
        self._tau = tau

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def axis(self) -> np.ndarray:
        return self._axis.copy()

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def effective_rabi(self) -> float:
        """Effective Rabi frequency including detuning."""
        n_z = self._axis[2]
        return np.sqrt(self._delta**2 + 2 * n_z * self._delta * self._omega + self._omega**2)

    def duration(self) -> float:
        return self._tau

    def hamiltonian(self) -> np.ndarray:
        """Return drive Hamiltonian in full space."""
        sx, sy, sz = self._subspace.subspace_pauli_matrices()
        n = self._axis
        H_drive = self._omega * (n[0] * sx + n[1] * sy + n[2] * sz) / 2
        H_detuning = self._delta * sz / 2
        return H_drive + H_detuning

    def unitary(self) -> np.ndarray:
        """Return evolution unitary."""
        return expm(-1j * self.hamiltonian() * self._tau)


# =============================================================================
# Multi-level pulse sequence
# =============================================================================

class MultiLevelPulseSequence:
    """
    Pulse sequence operating on a subspace of a multi-level system.

    This class generalizes pulse sequences to work on any 2-level subspace
    of a d-dimensional quantum system. The sequence tracks both the full
    system evolution and the effective 2x2 evolution on the subspace.

    Parameters
    ----------
    system : QuantumSystem
        The quantum system
    subspace : Subspace
        The 2-level subspace for the sequence

    Examples
    --------
    >>> system = ThreeLevelClock()
    >>> seq = MultiLevelPulseSequence(system, system.probe)
    >>> seq.add_instant_pulse([1, 0, 0], np.pi/2)  # pi/2 on probe
    >>> seq.add_free_evolution(1.0, delta=0.1)
    >>> seq.add_instant_pulse([1, 0, 0], np.pi/2)
    """

    def __init__(self, system: QuantumSystem, subspace: Subspace):
        self._system = system
        self._subspace = subspace
        self._elements: List[MultiLevelPulseElement] = []
        self._polynomials_computed = False
        self._poly_list = []
        self._polynomial_segments = []

    @property
    def system(self) -> QuantumSystem:
        return self._system

    @property
    def subspace(self) -> Subspace:
        return self._subspace

    @property
    def dim(self) -> int:
        return self._system.dim

    @property
    def elements(self) -> List[MultiLevelPulseElement]:
        return self._elements.copy()

    def add_element(self, element: MultiLevelPulseElement) -> 'MultiLevelPulseSequence':
        """Add a pulse element."""
        if element.subspace is not self._subspace:
            raise ValueError("Element subspace must match sequence subspace")
        self._elements.append(element)
        self._polynomials_computed = False
        return self

    def add_instant_pulse(self, axis: List[float], angle: float) -> 'MultiLevelPulseSequence':
        """Add an instantaneous pulse on the subspace."""
        element = MultiLevelInstantPulse(self._subspace, axis, angle)
        return self.add_element(element)

    def add_free_evolution(self, tau: float, delta: float = 0.0) -> 'MultiLevelPulseSequence':
        """Add free evolution on the subspace."""
        element = MultiLevelFreeEvolution(self._subspace, tau, delta)
        return self.add_element(element)

    def add_continuous_pulse(self, omega: float, axis: List[float],
                             delta: float, tau: float) -> 'MultiLevelPulseSequence':
        """Add a continuous pulse on the subspace."""
        element = MultiLevelContinuousPulse(self._subspace, omega, axis, delta, tau)
        return self.add_element(element)

    def total_duration(self) -> float:
        """Return total sequence duration."""
        return sum(e.duration() for e in self._elements)

    def total_unitary(self) -> np.ndarray:
        """Return total unitary in full space."""
        U = np.eye(self.dim, dtype=complex)
        for element in self._elements:
            U = element.unitary() @ U
        return U

    def total_subspace_unitary(self) -> np.ndarray:
        """Return total unitary projected to 2x2 subspace."""
        return self._subspace.project_operator(self.total_unitary())

    def compute_polynomials(self) -> List[Tuple]:
        """
        Compute QSP polynomial segments for the subspace evolution.

        The polynomials (f, g) satisfy:
            U_subspace = [[f, i*g], [i*conj(g), conj(f)]]
        with |f|^2 + |g|^2 = 1

        Returns
        -------
        list
            List of polynomial segment data for filter function computation
        """
        self._polynomial_segments = []
        self._poly_list = []

        curr_time = 0.0
        f_prev = 1.0 + 0j
        g_prev = 0.0 + 0j

        first_rotation = True

        for element in self._elements:
            if isinstance(element, MultiLevelFreeEvolution):
                delta = element.delta
                tau = element.tau
                theta = 0.0

                if first_rotation:
                    self._poly_list.append((0, 0, theta, tau, curr_time + tau))
                else:
                    self._poly_list.append((f_prev, g_prev, theta, tau, curr_time + tau))

                # Create polynomial functions
                def make_F(delta_val, t0_val, f_val):
                    def F(t):
                        return np.exp(1j * delta_val * (t - t0_val) / 2) * f_val
                    return F

                def make_G(delta_val, t0_val, g_val):
                    def G(t):
                        return np.exp(1j * delta_val * (t - t0_val) / 2) * g_val
                    return G

                start_time = curr_time
                end_time = curr_time + tau
                F_func = make_F(delta, curr_time, f_prev)
                G_func = make_G(delta, curr_time, g_prev)
                self._polynomial_segments.append((F_func, G_func, start_time, end_time))

                f_prev = F_func(end_time)
                g_prev = G_func(end_time)
                curr_time = end_time

            elif isinstance(element, MultiLevelInstantPulse):
                theta = element.angle
                cos_half = np.cos(theta / 2)
                sin_half = np.sin(theta / 2)

                # For x-axis rotation (generalize for arbitrary axis later)
                axis = element.axis
                if np.allclose(axis, [1, 0, 0]):
                    f_new = f_prev * cos_half - np.conj(g_prev) * sin_half
                    g_new = g_prev * cos_half + np.conj(f_prev) * sin_half
                elif np.allclose(axis, [0, 1, 0]):
                    # Y-axis rotation
                    f_new = f_prev * cos_half - 1j * np.conj(g_prev) * sin_half
                    g_new = g_prev * cos_half + 1j * np.conj(f_prev) * sin_half
                else:
                    # General rotation - use matrix formulation
                    U_2x2 = element.subspace_unitary()
                    f_new = U_2x2[0, 0] * f_prev + U_2x2[0, 1] / 1j * np.conj(g_prev)
                    g_new = U_2x2[1, 0] / 1j * np.conj(f_prev) + U_2x2[1, 1] * g_prev

                f_prev = f_new
                g_prev = g_new
                first_rotation = False

            elif isinstance(element, MultiLevelContinuousPulse):
                omega = element.omega
                axis = element.axis
                delta = element.delta
                tau = element.tau
                n_x, n_y, n_z = axis
                rabi = element.effective_rabi

                self._poly_list.append((f_prev, g_prev, omega, n_x, n_y, n_z, tau))

                # Compute end values using the continuous formula
                def make_F_continuous(rabi_val, delta_val, omega_val, axis_val,
                                       t0_val, f_val, g_val, first=False):
                    n_x, n_y, n_z = axis_val
                    def F(t):
                        dt = t - t0_val
                        if first:
                            return (np.cos(rabi_val * dt / 2) -
                                    1j * (delta_val + n_z * omega_val) / rabi_val *
                                    np.sin(rabi_val * dt / 2))
                        else:
                            term1 = ((np.cos(rabi_val * dt / 2) +
                                     1j * (delta_val + n_z * omega_val) / rabi_val *
                                     np.sin(rabi_val * dt / 2)) * f_val)
                            term2 = ((-n_x + 1j * n_y) * omega_val / rabi_val *
                                     np.sin(rabi_val * dt / 2) * np.conj(g_val))
                            return term1 + term2
                    return F

                def make_G_continuous(rabi_val, delta_val, omega_val, axis_val,
                                       t0_val, f_val, g_val, first=False):
                    n_x, n_y, n_z = axis_val
                    def G(t):
                        dt = t - t0_val
                        if first:
                            return ((n_x - 1j * n_y) * omega_val / rabi_val *
                                    np.sin(rabi_val * dt / 2))
                        else:
                            term1 = ((n_x - 1j * n_y) * omega_val / rabi_val *
                                     np.sin(rabi_val * dt / 2) * np.conj(f_val))
                            term2 = ((np.cos(rabi_val * dt / 2) +
                                     1j * (delta_val + n_z * omega_val) / rabi_val *
                                     np.sin(rabi_val * dt / 2)) * g_val)
                            return term1 + term2
                    return G

                start_time = curr_time
                end_time = curr_time + tau
                is_first = first_rotation
                F_func = make_F_continuous(rabi, delta, omega, axis, curr_time,
                                            f_prev, g_prev, first=is_first)
                G_func = make_G_continuous(rabi, delta, omega, axis, curr_time,
                                            f_prev, g_prev, first=is_first)

                self._polynomial_segments.append((F_func, G_func, start_time, end_time))

                f_prev = F_func(end_time)
                g_prev = G_func(end_time)
                curr_time = end_time
                first_rotation = False

        self._polynomials_computed = True
        return self._polynomial_segments

    def get_filter_function_calculator(self) -> 'SubspaceFilterFunction':
        """Return filter function calculator for this sequence."""
        if not self._polynomials_computed:
            self.compute_polynomials()

        # Determine sequence type
        has_continuous = any(isinstance(e, MultiLevelContinuousPulse)
                            for e in self._elements)

        return SubspaceFilterFunction(
            self._poly_list,
            self._subspace,
            continuous=has_continuous
        )

    def evolve_state(self, initial_state: np.ndarray) -> np.ndarray:
        """
        Evolve a state through the sequence.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial state vector (d-dimensional)

        Returns
        -------
        np.ndarray
            Final state vector
        """
        return self.total_unitary() @ initial_state


# =============================================================================
# Subspace filter function
# =============================================================================

class SubspaceFilterFunction(FilterFunction):
    """
    Filter function calculator for pulse sequences on subspaces.

    Computes the noise sensitivity for a pulse sequence operating on
    a 2-level subspace of a larger quantum system.

    Parameters
    ----------
    poly_list : list
        Polynomial segment data from MultiLevelPulseSequence
    subspace : Subspace
        The subspace on which the sequence operates
    continuous : bool
        Whether the sequence uses continuous pulses

    Notes
    -----
    The filter function describes sensitivity to noise operators.
    For a subspace {|i>, |j>} of a d-level system:

    - Dephasing noise: sigma_z^{ij} = |i><i| - |j><j|
    - Amplitude noise: sigma_x^{ij}, sigma_y^{ij}

    The filter function F(omega) satisfies:
        chi = integral |F(omega)|^2 S(omega) d omega / (2 pi)

    where chi is the decay parameter for the relevant coherence.
    """

    def __init__(self, poly_list: List[Tuple], subspace: Subspace,
                 continuous: bool = False):
        self._poly_list = poly_list
        self._subspace = subspace
        self._continuous = continuous

    @property
    def subspace(self) -> Subspace:
        return self._subspace

    def _compute_factor_instant(self, w: np.ndarray, t0: float,
                                 tau: float) -> np.ndarray:
        """Compute exponential factor for instantaneous pulses."""
        w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return 1j * np.exp(1j * w * t0) * (np.exp(-1j * w_safe * tau) - 1) / w_safe

    def filter_function(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute filter function components.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies

        Returns
        -------
        tuple
            (Fx, Fy, Fz) filter function components for the subspace
        """
        frequencies = np.asarray(frequencies)
        Fx_total = np.zeros_like(frequencies, dtype=complex)
        Fy_total = np.zeros_like(frequencies, dtype=complex)
        Fz_total = np.zeros_like(frequencies, dtype=complex)

        curr_time = 0.0

        if self._continuous:
            # Continuous pulse formulas
            for f, g, omega, n_x, n_y, n_z, tau in self._poly_list:
                phi = np.arctan2(-n_y, n_x) if (n_x != 0 or n_y != 0) else 0.0

                c_vals = np.array([cj(w, curr_time, omega, tau) for w in frequencies])
                s_vals = np.array([sj(w, curr_time, omega, tau) for w in frequencies])

                expr = (c_vals * f * np.conj(g) +
                        s_vals * (np.exp(1j * phi) * f**2 +
                                 np.exp(-1j * phi) * np.conj(g)**2))

                Fx_total += np.real(-1j * expr)
                Fy_total += np.imag(1j * expr)

                fz_expr = (c_vals * (f * np.conj(f) - g * np.conj(g)) -
                           s_vals * (np.exp(1j * phi) * f * g +
                                    np.exp(-1j * phi) * np.conj(f) * np.conj(g)))
                Fz_total += fz_expr

                curr_time += tau
        else:
            # Instantaneous pulse formulas
            for index, (f, g, theta, tau, t_end) in enumerate(self._poly_list):
                factor = self._compute_factor_instant(frequencies, curr_time, tau)

                if index == 0:
                    Fz_total += factor
                else:
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)

                    expr = 2 * cos_theta * f * g + sin_theta * (f**2 - np.conj(g)**2)

                    Fx_total += factor * np.real(1j * expr)
                    Fy_total += factor * np.imag(1j * expr)

                    fz_expr = (cos_theta * (f * np.conj(f) - g * np.conj(g)) -
                               sin_theta * (f * g + np.conj(f) * np.conj(g)))
                    Fz_total += factor * fz_expr

                curr_time = t_end - tau + tau

        return np.real(Fx_total), np.real(Fy_total), np.real(Fz_total)

    def filter_function_for_noise(self, frequencies: np.ndarray,
                                   noise_operator: np.ndarray) -> np.ndarray:
        """
        Compute filter function for a specific noise operator.

        This generalizes the filter function to arbitrary noise operators
        by projecting the noise onto the subspace Pauli basis.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies
        noise_operator : np.ndarray
            Noise operator in full space (d x d matrix)

        Returns
        -------
        np.ndarray
            Filter function |F(omega)|^2 for this noise
        """
        # Project noise operator onto subspace
        noise_2x2 = self._subspace.project_operator(noise_operator)

        # Decompose into Pauli components
        from .pulse_sequence import SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY

        # Pauli decomposition: M = a_0 I + a_x X + a_y Y + a_z Z
        a_x = np.real(np.trace(SIGMA_X @ noise_2x2)) / 2
        a_y = np.real(np.trace(SIGMA_Y @ noise_2x2)) / 2
        a_z = np.real(np.trace(SIGMA_Z @ noise_2x2)) / 2

        # Get filter function components
        Fx, Fy, Fz = self.filter_function(frequencies)

        # Total filter function for this noise
        return (a_x * Fx + a_y * Fy + a_z * Fz)**2


# =============================================================================
# Factory functions for common multi-level sequences
# =============================================================================

def multilevel_ramsey(system: QuantumSystem, subspace: Subspace,
                      tau: float, delta: float = 0.0) -> MultiLevelPulseSequence:
    """
    Create a Ramsey sequence on a subspace.

    Parameters
    ----------
    system : QuantumSystem
        The quantum system
    subspace : Subspace
        The 2-level subspace
    tau : float
        Free evolution time
    delta : float
        Detuning

    Returns
    -------
    MultiLevelPulseSequence
    """
    seq = MultiLevelPulseSequence(system, subspace)
    seq.add_instant_pulse([1, 0, 0], np.pi/2)
    seq.add_free_evolution(tau, delta)
    seq.add_instant_pulse([1, 0, 0], np.pi/2)
    return seq


def multilevel_spin_echo(system: QuantumSystem, subspace: Subspace,
                         tau: float, delta: float = 0.0) -> MultiLevelPulseSequence:
    """
    Create a spin echo sequence on a subspace.

    Parameters
    ----------
    system : QuantumSystem
        The quantum system
    subspace : Subspace
        The 2-level subspace
    tau : float
        Total free evolution time
    delta : float
        Detuning

    Returns
    -------
    MultiLevelPulseSequence
    """
    seq = MultiLevelPulseSequence(system, subspace)
    seq.add_instant_pulse([1, 0, 0], np.pi/2)
    seq.add_free_evolution(tau/2, delta)
    seq.add_instant_pulse([1, 0, 0], np.pi)
    seq.add_free_evolution(tau/2, delta)
    seq.add_instant_pulse([1, 0, 0], np.pi/2)
    return seq


def multilevel_cpmg(system: QuantumSystem, subspace: Subspace,
                    tau: float, n_pulses: int, delta: float = 0.0) -> MultiLevelPulseSequence:
    """
    Create a CPMG sequence on a subspace.

    Parameters
    ----------
    system : QuantumSystem
        The quantum system
    subspace : Subspace
        The 2-level subspace
    tau : float
        Total free evolution time
    n_pulses : int
        Number of pi pulses
    delta : float
        Detuning

    Returns
    -------
    MultiLevelPulseSequence
    """
    if n_pulses < 1:
        raise ValueError("n_pulses must be at least 1")

    interval = tau / (2 * n_pulses)

    seq = MultiLevelPulseSequence(system, subspace)
    seq.add_instant_pulse([1, 0, 0], np.pi/2)

    for _ in range(n_pulses):
        seq.add_free_evolution(interval, delta)
        seq.add_instant_pulse([0, 1, 0], np.pi)  # Y-axis for CPMG
        seq.add_free_evolution(interval, delta)

    seq.add_instant_pulse([1, 0, 0], np.pi/2)
    return seq


# =============================================================================
# Differential spectroscopy utilities
# =============================================================================

class DifferentialSpectroscopySequence:
    """
    Pulse sequence for differential (clock) spectroscopy.

    This implements the protocol where:
    1. System starts in clock superposition (|g> + |m>)/sqrt(2)
    2. QSP sequence acts on probe transition |g> <-> |e>
    3. Only |g> component acquires phase from probe
    4. Differential phase measured on clock transition

    Parameters
    ----------
    system : ThreeLevelClock
        The three-level clock system
    probe_sequence : MultiLevelPulseSequence
        Pulse sequence on the probe transition

    Notes
    -----
    The key advantage of differential spectroscopy is that common-mode
    noise (affecting both |g> and |m>) cancels out, leaving only the
    signal from the probe transition.
    """

    def __init__(self, system: 'QuantumSystem',
                 probe_sequence: MultiLevelPulseSequence):
        self._system = system
        self._probe_seq = probe_sequence

        # Verify we're using the probe subspace
        if hasattr(system, 'probe'):
            expected_subspace = system.probe
            if probe_sequence.subspace is not expected_subspace:
                raise ValueError("Sequence must operate on probe subspace")

    @property
    def system(self):
        return self._system

    @property
    def probe_sequence(self) -> MultiLevelPulseSequence:
        return self._probe_seq

    def compute_differential_phase(self, initial_clock_phase: float = 0.0) -> float:
        """
        Compute the differential phase accumulated.

        Parameters
        ----------
        initial_clock_phase : float
            Initial phase in clock superposition

        Returns
        -------
        float
            Differential phase shift
        """
        # Prepare clock superposition
        psi0 = self._system.prepare_clock_superposition(initial_clock_phase)

        # Evolve through probe sequence
        psi_final = self._probe_seq.evolve_state(psi0)

        # Extract clock phase
        return self._system.clock_phase(psi_final) - initial_clock_phase

    def differential_filter_function(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute filter function for differential measurement.

        The differential measurement is sensitive to probe dephasing
        but immune to global (common-mode) dephasing.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies

        Returns
        -------
        np.ndarray
            Differential filter function |F_diff(omega)|^2
        """
        ff = self._probe_seq.get_filter_function_calculator()

        # For differential measurement, we care about probe dephasing
        # which maps to the clock measurement
        Fx, Fy, Fz = ff.filter_function(frequencies)

        # The differential signal comes from Fz (dephasing on probe)
        # which creates relative phase between |g> and |m>
        return Fz**2
