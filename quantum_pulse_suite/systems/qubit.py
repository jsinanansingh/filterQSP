"""
Two-level (qubit) quantum system.

This provides backward compatibility with existing code while using
the new system abstraction.
"""

from typing import Dict, Optional
import numpy as np

from .base import QuantumSystem
from ..core.operators import SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY_2


class QubitSystem(QuantumSystem):
    """
    Standard two-level quantum system (qubit).

    This class wraps the existing qubit functionality in the new
    system abstraction, providing backward compatibility.

    Level structure:
        |0> = |g> (ground)
        |1> = |e> (excited)

    Parameters
    ----------
    level_labels : list of str, optional
        Labels for the levels (default: ['g', 'e'])

    Examples
    --------
    >>> system = QubitSystem()
    >>> system.dim
    2
    >>> system.sigma_x
    array([[0.+0.j, 1.+0.j],
           [1.+0.j, 0.+0.j]])
    """

    def __init__(self, level_labels: Optional[list] = None):
        if level_labels is None:
            level_labels = ['g', 'e']
        super().__init__(dim=2, level_labels=level_labels)

        # Register the single subspace (the whole system)
        self.register_subspace('qubit', (0, 1))

    @property
    def sigma_x(self) -> np.ndarray:
        """Pauli X matrix."""
        return SIGMA_X.copy()

    @property
    def sigma_y(self) -> np.ndarray:
        """Pauli Y matrix."""
        return SIGMA_Y.copy()

    @property
    def sigma_z(self) -> np.ndarray:
        """Pauli Z matrix."""
        return SIGMA_Z.copy()

    @property
    def sigma_plus(self) -> np.ndarray:
        """Raising operator |e><g|."""
        return np.array([[0, 0], [1, 0]], dtype=complex)

    @property
    def sigma_minus(self) -> np.ndarray:
        """Lowering operator |g><e|."""
        return np.array([[0, 1], [0, 0]], dtype=complex)

    def noise_operators(self) -> Dict[str, np.ndarray]:
        """
        Return standard qubit noise operators.

        Returns
        -------
        dict
            'dephasing': sigma_z (phase noise)
            'amplitude_x': sigma_x (transverse noise)
            'amplitude_y': sigma_y (transverse noise)
        """
        return {
            'dephasing': self.sigma_z,
            'amplitude_x': self.sigma_x,
            'amplitude_y': self.sigma_y,
        }

    def default_measurement_operator(self) -> np.ndarray:
        """
        Return default measurement operator (sigma_z).

        Returns
        -------
        np.ndarray
            Pauli Z matrix
        """
        return self.sigma_z

    def _prepare_named_state(self, name: str) -> np.ndarray:
        """Prepare a named state."""
        states = {
            'ground': np.array([1, 0], dtype=complex),
            'excited': np.array([0, 1], dtype=complex),
            'plus_x': np.array([1, 1], dtype=complex) / np.sqrt(2),
            'minus_x': np.array([1, -1], dtype=complex) / np.sqrt(2),
            'plus_y': np.array([1, 1j], dtype=complex) / np.sqrt(2),
            'minus_y': np.array([1, -1j], dtype=complex) / np.sqrt(2),
            'plus_z': np.array([1, 0], dtype=complex),
            'minus_z': np.array([0, 1], dtype=complex),
        }

        if name not in states:
            raise ValueError(f"Unknown state '{name}'. Available: {list(states.keys())}")

        return states[name]

    def bloch_vector(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Bloch vector for a pure state.

        Parameters
        ----------
        state : np.ndarray
            State vector (2-component)

        Returns
        -------
        np.ndarray
            Bloch vector [x, y, z]
        """
        state = np.asarray(state, dtype=complex)
        x = np.real(self.expectation_value(self.sigma_x, state))
        y = np.real(self.expectation_value(self.sigma_y, state))
        z = np.real(self.expectation_value(self.sigma_z, state))
        return np.array([x, y, z])
