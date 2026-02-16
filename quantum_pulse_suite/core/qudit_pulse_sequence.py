"""
Qudit pulse sequences built from spin-displacement pulses.

Provides pulse elements and sequence classes for d-level systems driven
by spin-displacement pulses, compatible with fft_filter_function().

Key classes:
- SpinDisplacementElement: A single finite-duration spin-displacement pulse
- InstantSpinDisplacement: Instantaneous limit of a spin-displacement pulse
- QuditFreeEvolution: Free evolution under Jz detuning
- QuditPulseSequence: Sequence of the above, feeds into FFT filter function
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Optional

from .spin_displacement import (
    spin_j_operators,
    spin_displacement_hamiltonian,
    spin_displacement_pulse,
    snap_gate,
)


class QuditPulseElement:
    """Base class for qudit pulse elements."""

    def __init__(self, d):
        self._d = d

    @property
    def d(self):
        return self._d

    def duration(self):
        raise NotImplementedError

    def hamiltonian(self):
        raise NotImplementedError

    def unitary(self):
        raise NotImplementedError


class SpinDisplacementElement(QuditPulseElement):
    """
    A finite-duration spin-displacement pulse.

    D(phases, theta) = expm(-i * tau * H_rot(phases) / omega)
    where tau = theta / omega is the pulse duration.

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    phases : array_like of length d-1, optional
        Tone phases. None = all zeros (Jx rotation).
    theta : float
        Rotation angle (pulse area).
    omega : float
        Rabi frequency, sets the timescale tau = theta/omega.
    detunings : array_like of length d-1, optional
        Transition detunings.
    """

    def __init__(self, d, phases=None, theta=np.pi, omega=1.0, detunings=None):
        super().__init__(d)
        self._phases = np.zeros(d - 1) if phases is None else np.asarray(phases, dtype=float)
        self._theta = theta
        self._omega = omega
        self._detunings = np.zeros(d - 1) if detunings is None else np.asarray(detunings, dtype=float)
        self._H = spin_displacement_hamiltonian(d, self._phases, omega, self._detunings)
        self._tau = abs(theta) / omega

    @property
    def phases(self):
        return self._phases.copy()

    @property
    def theta(self):
        return self._theta

    @property
    def omega(self):
        return self._omega

    @property
    def tau(self):
        return self._tau

    def duration(self):
        return self._tau

    def hamiltonian(self):
        return self._H.copy()

    def unitary(self):
        return expm(-1j * self._H * self._tau / self._omega)


class InstantSpinDisplacement(QuditPulseElement):
    """
    Instantaneous spin-displacement pulse (zero duration).

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    phases : array_like of length d-1, optional
        Tone phases. None = all zeros (Jx rotation).
    theta : float
        Rotation angle.
    detunings : array_like of length d-1, optional
        Transition detunings.
    """

    def __init__(self, d, phases=None, theta=np.pi, detunings=None):
        super().__init__(d)
        self._phases = np.zeros(d - 1) if phases is None else np.asarray(phases, dtype=float)
        self._theta = theta
        self._detunings = np.zeros(d - 1) if detunings is None else np.asarray(detunings, dtype=float)
        # Use omega=1 for Hamiltonian reference (arbitrary for instantaneous)
        self._U = spin_displacement_pulse(d, self._phases, theta, omega=1.0,
                                          detunings=self._detunings)

    @property
    def phases(self):
        return self._phases.copy()

    @property
    def theta(self):
        return self._theta

    def duration(self):
        return 0.0

    def hamiltonian(self):
        # Effective Hamiltonian (for reference; not used in propagation)
        return spin_displacement_hamiltonian(self._d, self._phases, 1.0,
                                            self._detunings)

    def unitary(self):
        return self._U.copy()


class QuditFreeEvolution(QuditPulseElement):
    """
    Free evolution under diagonal Hamiltonian (e.g., Jz detuning).

    H_free = delta * Jz, evolving for time tau.

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    tau : float
        Evolution duration.
    delta : float
        Detuning (coefficient of Jz). Default: 0.
    """

    def __init__(self, d, tau, delta=0.0):
        super().__init__(d)
        self._tau = tau
        self._delta = delta
        _, _, Jz = spin_j_operators(d)
        self._H = delta * Jz

    @property
    def tau(self):
        return self._tau

    @property
    def delta(self):
        return self._delta

    def duration(self):
        return self._tau

    def hamiltonian(self):
        return self._H.copy()

    def unitary(self):
        if abs(self._delta) < 1e-15:
            return np.eye(self._d, dtype=complex)
        return expm(-1j * self._H * self._tau)


class QuditPulseSequence:
    """
    Pulse sequence for a d-level system using spin-displacement pulses.

    Compatible with fft_filter_function() via .elements and .total_duration().

    Parameters
    ----------
    d : int
        Hilbert space dimension.

    Examples
    --------
    >>> seq = QuditPulseSequence(3)
    >>> seq.add_spin_displacement(theta=np.pi/2, omega=10.0)
    >>> seq.add_free_evolution(tau=1.0)
    >>> seq.add_spin_displacement(theta=np.pi/2, omega=10.0)
    """

    def __init__(self, d):
        self._d = d
        self._elements = []

    @property
    def d(self):
        return self._d

    @property
    def dim(self):
        """Alias for d, for compatibility."""
        return self._d

    @property
    def elements(self):
        return list(self._elements)

    def total_duration(self):
        return sum(e.duration() for e in self._elements)

    def total_unitary(self):
        U = np.eye(self._d, dtype=complex)
        for e in self._elements:
            U = e.unitary() @ U
        return U

    def add_element(self, element):
        """Add a pulse element to the sequence."""
        if element.d != self._d:
            raise ValueError(f"Element dimension {element.d} != sequence dimension {self._d}")
        self._elements.append(element)
        return self

    def add_spin_displacement(self, phases=None, theta=np.pi, omega=1.0,
                              detunings=None):
        """Add a finite-duration spin-displacement pulse."""
        elem = SpinDisplacementElement(self._d, phases, theta, omega, detunings)
        return self.add_element(elem)

    def add_instant_spin_displacement(self, phases=None, theta=np.pi,
                                      detunings=None):
        """Add an instantaneous spin-displacement pulse."""
        elem = InstantSpinDisplacement(self._d, phases, theta, detunings)
        return self.add_element(elem)

    def add_free_evolution(self, tau, delta=0.0):
        """Add free evolution under Jz detuning."""
        elem = QuditFreeEvolution(self._d, tau, delta)
        return self.add_element(elem)

    @classmethod
    def ramsey(cls, d, tau, delta=0.0, omega=1.0, continuous=True):
        """
        Create a qudit Ramsey sequence: pi/2 - free evolution - pi/2.

        Uses Jx rotations (phases=0) for the pi/2 pulses.

        Parameters
        ----------
        d : int
            Hilbert space dimension.
        tau : float
            Free evolution time.
        delta : float
            Detuning during free evolution.
        omega : float
            Rabi frequency for the pulses.
        continuous : bool
            If True, use finite-duration pulses. If False, instantaneous.
        """
        seq = cls(d)
        if continuous:
            seq.add_spin_displacement(theta=np.pi / 2, omega=omega)
            seq.add_free_evolution(tau, delta)
            seq.add_spin_displacement(theta=np.pi / 2, omega=omega)
        else:
            seq.add_instant_spin_displacement(theta=np.pi / 2)
            seq.add_free_evolution(tau, delta)
            seq.add_instant_spin_displacement(theta=np.pi / 2)
        return seq

    @classmethod
    def spin_echo(cls, d, tau, delta=0.0, omega=1.0, continuous=True):
        """
        Create a qudit spin echo: pi/2 - tau/2 - pi - tau/2 - pi/2.

        Uses Jx rotations (phases=0).
        """
        seq = cls(d)
        if continuous:
            seq.add_spin_displacement(theta=np.pi / 2, omega=omega)
            seq.add_free_evolution(tau / 2, delta)
            seq.add_spin_displacement(theta=np.pi, omega=omega)
            seq.add_free_evolution(tau / 2, delta)
            seq.add_spin_displacement(theta=np.pi / 2, omega=omega)
        else:
            seq.add_instant_spin_displacement(theta=np.pi / 2)
            seq.add_free_evolution(tau / 2, delta)
            seq.add_instant_spin_displacement(theta=np.pi)
            seq.add_free_evolution(tau / 2, delta)
            seq.add_instant_spin_displacement(theta=np.pi / 2)
        return seq

    @classmethod
    def cpmg(cls, d, tau, n_pulses, delta=0.0, omega=1.0, continuous=True):
        """
        Create a qudit CPMG sequence with Jx rotations.

        pi/2 - [tau/(2n) - pi - tau/(2n)]^n - pi/2
        """
        if n_pulses < 1:
            raise ValueError("n_pulses must be >= 1")

        interval = tau / (2 * n_pulses)
        seq = cls(d)

        if continuous:
            seq.add_spin_displacement(theta=np.pi / 2, omega=omega)
            for _ in range(n_pulses):
                seq.add_free_evolution(interval, delta)
                seq.add_spin_displacement(theta=np.pi, omega=omega)
                seq.add_free_evolution(interval, delta)
            seq.add_spin_displacement(theta=np.pi / 2, omega=omega)
        else:
            seq.add_instant_spin_displacement(theta=np.pi / 2)
            for _ in range(n_pulses):
                seq.add_free_evolution(interval, delta)
                seq.add_instant_spin_displacement(theta=np.pi)
                seq.add_free_evolution(interval, delta)
            seq.add_instant_spin_displacement(theta=np.pi / 2)
        return seq

    @classmethod
    def from_pulse_params(cls, d, phases_list, thetas, omega=1.0,
                          detunings_list=None, continuous=True):
        """
        Build a sequence from lists of pulse parameters.

        This is the interface for importing from MQS Prog or other optimizers.

        Parameters
        ----------
        d : int
            Hilbert space dimension.
        phases_list : list of array_like
            Phases for each pulse, each of length d-1.
        thetas : list of float
            Rotation angles for each pulse.
        omega : float
            Rabi frequency.
        detunings_list : list of array_like, optional
            Detunings for each pulse.
        continuous : bool
            If True, finite-duration pulses. If False, instantaneous.

        Returns
        -------
        QuditPulseSequence
        """
        n_pulses = len(thetas)
        if len(phases_list) != n_pulses:
            raise ValueError("phases_list and thetas must have same length")

        seq = cls(d)
        for n in range(n_pulses):
            dets = detunings_list[n] if detunings_list is not None else None
            if continuous:
                seq.add_spin_displacement(phases_list[n], thetas[n], omega, dets)
            else:
                seq.add_instant_spin_displacement(phases_list[n], thetas[n], dets)
        return seq
