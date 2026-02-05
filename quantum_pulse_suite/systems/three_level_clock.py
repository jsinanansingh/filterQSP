"""
Three-level clock system for differential spectroscopy.

This module implements a 3-level system suitable for global phase spectroscopy
experiments, such as those described in Zaporski et al.

Level Structure
---------------
    |2> = |m>  (metastable/clock state)
     |
    |0> = |g>  (ground state) <---> |1> = |e> (excited/probe state)

Transitions:
- Probe transition: |g> <-> |e> (levels 0-1) - QSP pulse sequences act here
- Clock transition: |g> <-> |m> (levels 0-2) - Measurement/readout here

Typical Experiment Protocol:
1. Prepare superposition (|g> + |m>)/sqrt(2) in clock subspace
2. Apply QSP sequence on probe transition |g> <-> |e>
3. The |g> component acquires phase relative to |m>
4. Measure on clock transition to read out phase

References
----------
- Zaporski et al., "Global Phase Spectroscopy" (for the experimental context)
"""

from typing import Dict, Optional, Tuple
import numpy as np

from .base import QuantumSystem
from ..core.hilbert_space import Subspace
from ..core.operators import (
    x_operator, y_operator, z_operator,
    transition_matrix, subspace_pauli
)


class ThreeLevelClock(QuantumSystem):
    """
    Three-level system for differential clock spectroscopy.

    Level structure:
        Level 0: |g> - ground state (shared between both transitions)
        Level 1: |e> - excited/probe state
        Level 2: |m> - metastable/clock state

    Two subspaces are automatically registered:
        'probe': levels (0, 1) = |g> <-> |e>
        'clock': levels (0, 2) = |g> <-> |m>

    Parameters
    ----------
    level_labels : list of str, optional
        Labels for the three levels (default: ['g', 'e', 'm'])

    Examples
    --------
    >>> system = ThreeLevelClock()
    >>> system.probe  # Subspace for probe transition
    Subspace('probe': |g> <-> |e>)
    >>> system.clock  # Subspace for clock transition
    Subspace('clock': |g> <-> |m>)

    # Prepare clock superposition
    >>> psi = system.prepare_clock_superposition()
    >>> print(psi)  # (|g> + |m>)/sqrt(2)
    [0.707+0.j 0.   +0.j 0.707+0.j]

    # Get Pauli matrices for probe transition
    >>> sx, sy, sz = system.probe_pauli_matrices()
    """

    # Level indices
    GROUND = 0
    EXCITED = 1
    METASTABLE = 2

    def __init__(self, level_labels: Optional[list] = None):
        if level_labels is None:
            level_labels = ['g', 'e', 'm']

        super().__init__(dim=3, level_labels=level_labels)

        # Register the two transition subspaces
        self._probe = self.register_subspace('probe', (self.GROUND, self.EXCITED))
        self._clock = self.register_subspace('clock', (self.GROUND, self.METASTABLE))

    @property
    def probe(self) -> Subspace:
        """The probe transition subspace |g> <-> |e>."""
        return self._probe

    @property
    def clock(self) -> Subspace:
        """The clock transition subspace |g> <-> |m>."""
        return self._clock

    # =========================================================================
    # Pauli-like operators for each subspace
    # =========================================================================

    def probe_pauli_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return Pauli matrices for the probe transition.

        Returns
        -------
        tuple
            (sigma_x, sigma_y, sigma_z) as 3x3 matrices acting on |g> <-> |e>
        """
        return subspace_pauli(3, (self.GROUND, self.EXCITED))

    def clock_pauli_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return Pauli matrices for the clock transition.

        Returns
        -------
        tuple
            (sigma_x, sigma_y, sigma_z) as 3x3 matrices acting on |g> <-> |m>
        """
        return subspace_pauli(3, (self.GROUND, self.METASTABLE))

    # =========================================================================
    # State preparation
    # =========================================================================

    def prepare_clock_superposition(self, phase: float = 0.0) -> np.ndarray:
        """
        Prepare the standard clock superposition state.

        Returns (|g> + exp(i*phase)|m>)/sqrt(2)

        Parameters
        ----------
        phase : float
            Relative phase between |g> and |m> (default: 0)

        Returns
        -------
        np.ndarray
            3-component state vector
        """
        state = np.zeros(3, dtype=complex)
        state[self.GROUND] = 1.0 / np.sqrt(2)
        state[self.METASTABLE] = np.exp(1j * phase) / np.sqrt(2)
        return state

    def prepare_probe_superposition(self, phase: float = 0.0) -> np.ndarray:
        """
        Prepare superposition in probe subspace.

        Returns (|g> + exp(i*phase)|e>)/sqrt(2)

        Parameters
        ----------
        phase : float
            Relative phase

        Returns
        -------
        np.ndarray
            3-component state vector
        """
        state = np.zeros(3, dtype=complex)
        state[self.GROUND] = 1.0 / np.sqrt(2)
        state[self.EXCITED] = np.exp(1j * phase) / np.sqrt(2)
        return state

    def _prepare_named_state(self, name: str) -> np.ndarray:
        """Prepare a named state."""
        if name in ['ground', 'g', '0']:
            return self.hilbert_space.basis_state(self.GROUND)
        elif name in ['excited', 'e', '1']:
            return self.hilbert_space.basis_state(self.EXCITED)
        elif name in ['metastable', 'm', '2']:
            return self.hilbert_space.basis_state(self.METASTABLE)
        elif name in ['clock_superposition', 'clock+']:
            return self.prepare_clock_superposition()
        elif name in ['probe_superposition', 'probe+']:
            return self.prepare_probe_superposition()
        else:
            raise ValueError(f"Unknown state '{name}'")

    # =========================================================================
    # Noise operators
    # =========================================================================

    def noise_operators(self) -> Dict[str, np.ndarray]:
        """
        Return noise operators for this system.

        Returns
        -------
        dict
            Noise operators for different channels:
            - 'probe_dephasing': dephasing on probe transition
            - 'clock_dephasing': dephasing on clock transition
            - 'probe_amplitude': amplitude noise on probe
            - 'global_dephasing': dephasing of |g> relative to |e> and |m>
        """
        sx_p, sy_p, sz_p = self.probe_pauli_matrices()
        sx_c, sy_c, sz_c = self.clock_pauli_matrices()

        return {
            'probe_dephasing': sz_p,
            'clock_dephasing': sz_c,
            'probe_amplitude_x': sx_p,
            'probe_amplitude_y': sy_p,
            'global_dephasing': self.global_dephasing_operator(),
        }

    def global_dephasing_operator(self) -> np.ndarray:
        """
        Return the global dephasing operator.

        This represents noise that shifts the energy of |g> relative to
        both |e> and |m>. In differential spectroscopy, this common-mode
        noise is often what limits sensitivity.

        Returns
        -------
        np.ndarray
            Operator proportional to 2|g><g| - |e><e| - |m><m|
        """
        op = np.zeros((3, 3), dtype=complex)
        op[self.GROUND, self.GROUND] = 2
        op[self.EXCITED, self.EXCITED] = -1
        op[self.METASTABLE, self.METASTABLE] = -1
        return op

    def differential_phase_operator(self) -> np.ndarray:
        """
        Return the differential phase operator for clock measurement.

        This measures the phase accumulated by |g> relative to |m>,
        which is the signal in differential spectroscopy.

        Returns
        -------
        np.ndarray
            sigma_z on clock subspace: |g><g| - |m><m|
        """
        _, _, sz_c = self.clock_pauli_matrices()
        return sz_c

    # =========================================================================
    # Measurement operators
    # =========================================================================

    def default_measurement_operator(self) -> np.ndarray:
        """
        Return default measurement operator (clock sigma_z).

        Returns
        -------
        np.ndarray
            Differential phase operator
        """
        return self.differential_phase_operator()

    def population_measurement(self, level: int) -> np.ndarray:
        """
        Return population measurement operator for a specific level.

        In experiments, this corresponds to shelving + fluorescence detection
        or similar level-selective readout.

        Parameters
        ----------
        level : int
            Which level to measure (0, 1, or 2)

        Returns
        -------
        np.ndarray
            Projector |level><level|
        """
        return self.projector(level)

    # =========================================================================
    # Hamiltonian construction
    # =========================================================================

    def probe_hamiltonian(self, omega: float, axis: np.ndarray,
                          delta: float = 0.0) -> np.ndarray:
        """
        Construct Hamiltonian for driving the probe transition.

        H = (omega/2) * (axis . sigma_probe) + (delta/2) * sigma_z_probe

        Parameters
        ----------
        omega : float
            Rabi frequency
        axis : np.ndarray
            Driving axis [x, y, z] (normalized)
        delta : float
            Detuning from resonance

        Returns
        -------
        np.ndarray
            3x3 Hamiltonian matrix
        """
        sx, sy, sz = self.probe_pauli_matrices()

        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        H = (omega / 2) * (axis[0] * sx + axis[1] * sy + axis[2] * sz)
        H += (delta / 2) * sz

        return H

    def clock_hamiltonian(self, omega: float, axis: np.ndarray,
                          delta: float = 0.0) -> np.ndarray:
        """
        Construct Hamiltonian for driving the clock transition.

        H = (omega/2) * (axis . sigma_clock) + (delta/2) * sigma_z_clock

        Parameters
        ----------
        omega : float
            Rabi frequency
        axis : np.ndarray
            Driving axis [x, y, z]
        delta : float
            Detuning

        Returns
        -------
        np.ndarray
            3x3 Hamiltonian matrix
        """
        sx, sy, sz = self.clock_pauli_matrices()

        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        H = (omega / 2) * (axis[0] * sx + axis[1] * sy + axis[2] * sz)
        H += (delta / 2) * sz

        return H

    def free_evolution_hamiltonian(self, delta_probe: float = 0.0,
                                    delta_clock: float = 0.0) -> np.ndarray:
        """
        Construct free evolution Hamiltonian.

        H = (delta_probe/2) * sigma_z_probe + (delta_clock/2) * sigma_z_clock

        Parameters
        ----------
        delta_probe : float
            Detuning on probe transition
        delta_clock : float
            Detuning on clock transition

        Returns
        -------
        np.ndarray
            3x3 Hamiltonian matrix
        """
        _, _, sz_p = self.probe_pauli_matrices()
        _, _, sz_c = self.clock_pauli_matrices()

        return (delta_probe / 2) * sz_p + (delta_clock / 2) * sz_c

    # =========================================================================
    # Utility methods
    # =========================================================================

    def clock_coherence(self, state: np.ndarray) -> complex:
        """
        Extract the clock coherence <g|rho|m> from a pure state.

        Parameters
        ----------
        state : np.ndarray
            3-component state vector

        Returns
        -------
        complex
            The coherence element rho_gm = state[g] * conj(state[m])
        """
        return state[self.GROUND] * np.conj(state[self.METASTABLE])

    def clock_phase(self, state: np.ndarray) -> float:
        """
        Extract the accumulated clock phase.

        Parameters
        ----------
        state : np.ndarray
            3-component state vector

        Returns
        -------
        float
            Phase angle of the g-m coherence
        """
        coherence = self.clock_coherence(state)
        return np.angle(coherence)

    def contrast(self, state: np.ndarray) -> float:
        """
        Compute the clock measurement contrast.

        Contrast = 2 * |<g|rho|m>| = 2 * |rho_gm|

        For a perfect superposition, contrast = 1.

        Parameters
        ----------
        state : np.ndarray
            3-component state vector

        Returns
        -------
        float
            Contrast in [0, 1]
        """
        return 2 * np.abs(self.clock_coherence(state))
