"""
Pulse sequence classes for quantum signal processing.

This module provides abstract base classes and concrete implementations for
instantaneous and continuous quantum pulse sequences.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable
import numpy as np
from scipy.linalg import expm

from quantum_pulse_suite.core.filter_functions import FilterFunction

# Pauli matrices (using convention R_x(t) = exp(i * sigma_x * t / 2))
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def normalize_axis(axis: List[float]) -> np.ndarray:
    """Normalize a 3D axis vector."""
    axis_array = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_array)
    if norm < 1e-12:
        raise ValueError("Axis vector cannot be zero.")
    return axis_array / norm


def axis_from_spherical(theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian unit vector.

    Parameters
    ----------
    theta : float
        Polar angle in radians (from z-axis)
    phi : float
        Azimuthal angle in radians (from x-axis in xy-plane)

    Returns
    -------
    np.ndarray
        Unit vector [x, y, z]
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


class PulseElement(ABC):
    """Abstract base class for pulse elements in a sequence."""

    @abstractmethod
    def duration(self) -> float:
        """Return the duration of this pulse element."""
        pass

    @abstractmethod
    def hamiltonian(self) -> np.ndarray:
        """Return the Hamiltonian for this pulse element."""
        pass

    @abstractmethod
    def unitary(self) -> np.ndarray:
        """Return the unitary evolution operator for this pulse element."""
        pass


class InstantaneousPulse(PulseElement):
    """
    An instantaneous rotation pulse.

    Parameters
    ----------
    axis : array-like
        Rotation axis as [x, y, z] (will be normalized)
    angle : float
        Rotation angle in radians
    """

    def __init__(self, axis: List[float], angle: float):
        self._axis = normalize_axis(axis)
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
        """Return effective Hamiltonian (infinite amplitude, zero duration)."""
        n = self._axis
        return (n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z) / 2

    def unitary(self) -> np.ndarray:
        """Return the rotation unitary exp(-i * n·σ * θ / 2)."""
        n = self._axis
        H = (n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z) / 2
        return expm(-1j * H * self._angle)


class ContinuousPulse(PulseElement):
    """
    A continuous (finite-duration) pulse.

    Parameters
    ----------
    omega : float
        Rabi frequency (rotation amplitude)
    axis : array-like
        Rotation axis as [x, y, z] (will be normalized)
    delta : float
        Detuning from resonance
    tau : float
        Duration of the pulse
    """

    def __init__(self, omega: float, axis: List[float], delta: float, tau: float):
        self._omega = omega
        self._axis = normalize_axis(axis)
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
        """Return the Hamiltonian for this pulse."""
        n = self._axis
        H_drive = self._omega * (n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z) / 2
        H_detuning = self._delta * SIGMA_Z / 2
        return H_drive + H_detuning

    def unitary(self) -> np.ndarray:
        """Return the unitary evolution operator."""
        return expm(-1j * self.hamiltonian() * self._tau)


class FreeEvolution(PulseElement):
    """
    Free evolution under detuning only.

    Parameters
    ----------
    tau : float
        Duration of free evolution
    delta : float
        Detuning (frequency offset)
    """

    def __init__(self, tau: float, delta: float = 0.0):
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
        """Return the free evolution Hamiltonian."""
        return self._delta * SIGMA_Z / 2

    def unitary(self) -> np.ndarray:
        """Return the unitary evolution operator."""
        return expm(-1j * self.hamiltonian() * self._tau)


class PulseSequence(ABC):
    """Abstract base class for pulse sequences."""

    def __init__(self):
        self._elements: List[PulseElement] = []
        self._polynomials_computed = False
        self._polynomial_segments = []

    def add_element(self, element: PulseElement) -> 'PulseSequence':
        """Add a pulse element to the sequence."""
        self._elements.append(element)
        self._polynomials_computed = False
        return self

    @property
    def elements(self) -> List[PulseElement]:
        """Return a copy of the elements list."""
        return self._elements.copy()

    def total_duration(self) -> float:
        """Return the total duration of the sequence."""
        return sum(e.duration() for e in self._elements)

    def total_unitary(self) -> np.ndarray:
        """Return the total unitary evolution operator."""
        U = IDENTITY.copy()
        for element in self._elements:
            U = element.unitary() @ U
        return U

    @abstractmethod
    def compute_polynomials(self) -> List[Tuple]:
        """
        Compute the QSP polynomial segments.

        Returns
        -------
        list
            List of polynomial segment tuples
        """
        pass

    def unitary_at_time(self, t: float) -> np.ndarray:
        """
        Compute the unitary at a specific time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the unitary

        Returns
        -------
        np.ndarray
            2x2 unitary matrix at time t
        """
        if not self._polynomials_computed:
            self.compute_polynomials()

        for F, G, start_time, end_time in self._polynomial_segments:
            if start_time <= t < end_time or (t == end_time and end_time == self.total_duration()):
                f_val = F(t)
                g_val = G(t)
                return np.array([
                    [f_val, 1j * g_val],
                    [1j * np.conj(g_val), np.conj(f_val)]
                ])

        raise ValueError(f"Time {t} is outside the sequence duration [0, {self.total_duration()})")

    def unitary_trajectory(self, times: np.ndarray) -> np.ndarray:
        """
        Compute the unitary trajectory at specified times.

        Parameters
        ----------
        times : np.ndarray
            Array of time points

        Returns
        -------
        np.ndarray
            Array of shape (len(times), 2, 2) containing unitaries
        """
        if not self._polynomials_computed:
            self.compute_polynomials()

        U_history = []
        for t in times:
            U_history.append(self.unitary_at_time(t))
        return np.array(U_history)

    @abstractmethod
    def get_filter_function_calculator(self) -> 'FilterFunction':
        """Return a filter function calculator for this sequence."""
        pass


class InstantaneousPulseSequence(PulseSequence):
    """
    Pulse sequence with instantaneous pulses and free evolutions.

    This represents the 'instant equiangular' QSP convention where
    x-rotations are instantaneous and interleaved with z-rotations
    (free evolutions under detuning).
    """

    def __init__(self):
        super().__init__()

    def add_free_evolution(self, tau: float, delta: float = 0.0) -> 'InstantaneousPulseSequence':
        """Add a free evolution segment."""
        self.add_element(FreeEvolution(tau, delta))
        return self

    def add_instant_pulse(self, axis: List[float], angle: float) -> 'InstantaneousPulseSequence':
        """Add an instantaneous pulse."""
        self.add_element(InstantaneousPulse(axis, angle))
        return self

    def compute_polynomials(self) -> List[Tuple]:
        """
        Compute polynomial segments for instant equiangular QSP.

        The polynomials follow the recurrence:
        F(t) = exp(i*delta*(t-t0)/2) * (f_prev*cos(theta/2) - g_prev*sin(theta/2))
        G(t) = exp(i*delta*(t-t0)/2) * (g_prev*cos(theta/2) + f_prev*sin(theta/2))

        Returns
        -------
        list
            List of (F, G, start_time, end_time) tuples
        """
        self._polynomial_segments = []
        self._poly_list = []  # For filter function computation

        curr_time = 0.0
        f_prev = 1.0
        g_prev = 0.0
        theta = 0.0

        # Track whether we've seen an x-rotation yet
        first_rotation = True

        for element in self._elements:
            if isinstance(element, FreeEvolution):
                delta = element.delta
                tau = element.tau
                theta = 0.0  # No rotation during free evolution

                # Store polynomial info for filter functions
                if first_rotation:
                    theta = 0.0
                    self._poly_list.append((0, 0, theta, tau, curr_time + tau))
                else:
                    self._poly_list.append((f_prev, g_prev, theta, tau, curr_time + tau))

                # Create polynomial functions for this segment using function factories
                def make_F(delta, tau, t0, f_prev):
                    def F(t):
                        return np.exp(1j * delta * (t - t0) / 2) * f_prev
                    return F

                def make_G(delta, tau, t0, g_prev):
                    def G(t):
                        return np.exp(1j * delta * (t - t0) / 2) * g_prev
                    return G
                
                start_time = curr_time
                end_time = curr_time + tau
                F_func = make_F(delta, tau, curr_time, f_prev)
                G_func = make_G(delta, tau, curr_time, g_prev)
                self._polynomial_segments.append((F_func, G_func, start_time, end_time))

                # Update for next segment - evaluate at end of this segment
                f_prev = F_func(end_time)
                g_prev = G_func(end_time)
                curr_time = end_time

            elif isinstance(element, InstantaneousPulse):
                # Apply rotation transformation to polynomial coefficients
                # Assuming x-axis rotation: R_x(theta)
                theta = element.angle
                cos_half = np.cos(theta / 2)
                sin_half = np.sin(theta / 2)

                f_new = f_prev * cos_half - np.conj(g_prev) * sin_half
                g_new = g_prev * cos_half + np.conj(f_prev) * sin_half

                f_prev = f_new
                g_prev = g_new
                first_rotation = False

        self._polynomials_computed = True
        return self._polynomial_segments

    def get_filter_function_calculator(self):
        """Return filter function calculator for instantaneous sequences."""
        if not self._polynomials_computed:
            self.compute_polynomials()

        from .filter_functions import InstantaneousFilterFunction
        return InstantaneousFilterFunction(self._poly_list)


class ContinuousPulseSequence(PulseSequence):
    """
    Pulse sequence with continuous (finite-duration) pulses.

    This represents sequences where pulses have finite Rabi frequency
    and duration, including detuning effects.
    """

    def __init__(self):
        super().__init__()

    def add_continuous_pulse(self, omega: float, axis: List[float],
                             delta: float, tau: float) -> 'ContinuousPulseSequence':
        """Add a continuous pulse."""
        self.add_element(ContinuousPulse(omega, axis, delta, tau))
        return self

    def compute_polynomials(self) -> List[Tuple]:
        """
        Compute polynomial segments for continuous QSP.

        Uses the equiXYZ formulas with effective Rabi frequency.

        Returns
        -------
        list
            List of (F, G, start_time, end_time) tuples
        """
        self._polynomial_segments = []
        self._poly_list = []  # For filter function computation

        curr_time = 0.0
        f_prev_val = 1.0
        g_prev_val = 0.0

        for i, element in enumerate(self._elements):
            if not isinstance(element, ContinuousPulse):
                continue

            omega = element.omega
            axis = element.axis
            delta = element.delta
            tau = element.tau

            n_x, n_y, n_z = axis
            rabi = element.effective_rabi

            # Store for filter function calculation
            self._poly_list.append((f_prev_val, g_prev_val, omega, n_x, n_y, n_z, tau))

            if i == 0:
                def make_F_first(rabi=rabi, delta=delta, omega=omega, axis=axis,
                                 tau=tau, t0=curr_time):
                    n_z = axis[2]
                    def F(t):
                        dt = t - t0
                        return (np.cos(rabi * dt / 2) -
                                1j * (delta + n_z * omega) / rabi * np.sin(rabi * dt / 2))
                    return F

                def make_G_first(rabi=rabi, delta=delta, omega=omega, axis=axis,
                                 tau=tau, t0=curr_time):
                    n_x, n_y = axis[0], axis[1]
                    def G(t):
                        dt = t - t0
                        return (n_x - 1j * n_y) * omega / rabi * np.sin(rabi * dt / 2)
                    return G

                F_func = make_F_first(rabi, delta, omega, axis, tau, curr_time)
                G_func = make_G_first(rabi, delta, omega, axis, tau, curr_time)
            else:
                def make_F_subsequent(rabi=rabi, delta=delta, omega=omega, axis=axis, tau=tau,
                           t0=curr_time, f_prev=f_prev_val, g_prev=g_prev_val):
                    n_x, n_y, n_z = axis
                    def F(t):
                        dt = t - t0
                        term1 = ((np.cos(rabi * dt / 2) +
                                 1j * (delta + n_z * omega) / rabi * np.sin(rabi * dt / 2)) *
                                 f_prev)
                        term2 = ((-n_x + 1j * n_y) * omega / rabi *
                                 np.sin(rabi * dt / 2) * np.conj(g_prev))
                        return term1 + term2
                    return F

                def make_G_subsequent(rabi=rabi, delta=delta, omega=omega, axis=axis, tau=tau,
                           t0=curr_time, f_prev=f_prev_val, g_prev=g_prev_val):
                    n_x, n_y, n_z = axis
                    def G(t):
                        dt = t - t0
                        term1 = ((n_x - 1j * n_y) * omega / rabi *
                                 np.sin(rabi * dt / 2) * np.conj(f_prev))
                        term2 = ((np.cos(rabi * dt / 2) +
                                 1j * (delta + n_z * omega) / rabi * np.sin(rabi * dt / 2)) *
                                 g_prev)
                        return term1 + term2
                    return G

                F = make_F_subsequent()
                G = make_G_subsequent()

            start_time = curr_time
            end_time = curr_time + tau
            self._polynomial_segments.append((F, G, start_time, end_time))

            # Update for next segment
            f_prev_val = F(end_time)
            g_prev_val = G(end_time)
            curr_time = end_time

        self._polynomials_computed = True
        return self._polynomial_segments

    def get_filter_function_calculator(self):
        """Return filter function calculator for continuous sequences."""
        if not self._polynomials_computed:
            self.compute_polynomials()

        from .filter_functions import ContinuousFilterFunction
        return ContinuousFilterFunction(self._poly_list)


# Factory functions for common sequences

def ramsey_sequence(tau: float, delta: float = 0.0) -> InstantaneousPulseSequence:
    """
    Create a Ramsey sequence: π/2 - τ - π/2

    Parameters
    ----------
    tau : float
        Free evolution time between pulses
    delta : float
        Detuning during free evolution

    Returns
    -------
    InstantaneousPulseSequence
        The Ramsey sequence
    """
    seq = InstantaneousPulseSequence()
    seq.add_instant_pulse([1, 0, 0], np.pi/2)  # π/2 x-rotation
    seq.add_free_evolution(tau, delta)
    seq.add_instant_pulse([1, 0, 0], np.pi/2)  # π/2 x-rotation
    return seq


def spin_echo_sequence(tau: float, delta: float = 0.0) -> InstantaneousPulseSequence:
    """
    Create a Hahn spin echo sequence: π/2 - τ/2 - π - τ/2 - π/2

    Parameters
    ----------
    tau : float
        Total free evolution time
    delta : float
        Detuning during free evolution

    Returns
    -------
    InstantaneousPulseSequence
        The spin echo sequence
    """
    seq = InstantaneousPulseSequence()
    seq.add_instant_pulse([1, 0, 0], np.pi/2)  # π/2 x-rotation
    seq.add_free_evolution(tau/2, delta)
    seq.add_instant_pulse([1, 0, 0], np.pi)    # π x-rotation (refocusing)
    seq.add_free_evolution(tau/2, delta)
    seq.add_instant_pulse([1, 0, 0], np.pi/2)  # π/2 x-rotation
    return seq


def cpmg_sequence(tau: float, n_pulses: int, delta: float = 0.0) -> InstantaneousPulseSequence:
    """
    Create a CPMG (Carr-Purcell-Meiboom-Gill) sequence.

    Structure: π/2 - (τ/2n - π - τ/2n)×n - π/2

    Parameters
    ----------
    tau : float
        Total free evolution time
    n_pulses : int
        Number of π pulses
    delta : float
        Detuning during free evolution

    Returns
    -------
    InstantaneousPulseSequence
        The CPMG sequence
    """
    if n_pulses < 1:
        raise ValueError("n_pulses must be at least 1")

    interval = tau / (2 * n_pulses)

    seq = InstantaneousPulseSequence()
    seq.add_instant_pulse([1, 0, 0], np.pi/2)  # Initial π/2 x-rotation

    for _ in range(n_pulses):
        seq.add_free_evolution(interval, delta)
        seq.add_instant_pulse([0, 1, 0], np.pi)  # π y-rotation (CPMG uses y-axis)
        seq.add_free_evolution(interval, delta)

    seq.add_instant_pulse([1, 0, 0], np.pi/2)  # Final π/2 x-rotation
    return seq


def continuous_rabi_sequence(omega: float, tau: float, axis: List[float] = None,
                              delta: float = 0.0) -> ContinuousPulseSequence:
    """
    Create a single continuous Rabi pulse.

    Parameters
    ----------
    omega : float
        Rabi frequency
    tau : float
        Pulse duration
    axis : list, optional
        Rotation axis, defaults to [1, 0, 0] (x-axis)
    delta : float
        Detuning

    Returns
    -------
    ContinuousPulseSequence
        Single pulse sequence
    """
    if axis is None:
        axis = [1, 0, 0]

    seq = ContinuousPulseSequence()
    seq.add_continuous_pulse(omega, axis, delta, tau)
    return seq
