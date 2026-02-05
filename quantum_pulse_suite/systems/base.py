"""
Abstract base class for quantum systems.

This module defines the interface that all quantum system implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

from ..core.hilbert_space import HilbertSpace, Subspace, LiouvilleSpace


class QuantumSystem(ABC):
    """
    Abstract base class for quantum systems of any dimension.

    Subclasses must implement methods for:
    - Defining the Hilbert space structure
    - Specifying noise channels
    - Computing filter functions

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space
    """

    def __init__(self, dim: int, level_labels: Optional[List[str]] = None):
        self._hilbert = HilbertSpace(dim, level_labels)
        self._liouville = LiouvilleSpace(self._hilbert)
        self._subspaces: Dict[str, Subspace] = {}

    @property
    def dim(self) -> int:
        """Dimension of the Hilbert space."""
        return self._hilbert.dim

    @property
    def hilbert_space(self) -> HilbertSpace:
        """The Hilbert space for this system."""
        return self._hilbert

    @property
    def liouville_space(self) -> LiouvilleSpace:
        """The Liouville space for this system."""
        return self._liouville

    @property
    def identity(self) -> np.ndarray:
        """Identity operator."""
        return self._hilbert.identity

    def register_subspace(self, name: str, levels: Tuple[int, int]) -> Subspace:
        """
        Register a named 2-level subspace.

        Parameters
        ----------
        name : str
            Name for the subspace (e.g., 'optical', 'metastable')
        levels : tuple of int
            The two levels defining the subspace

        Returns
        -------
        Subspace
            The created subspace object
        """
        subspace = Subspace(self._hilbert, levels, label=name)
        self._subspaces[name] = subspace
        return subspace

    def get_subspace(self, name: str) -> Subspace:
        """
        Get a registered subspace by name.

        Parameters
        ----------
        name : str
            Name of the subspace

        Returns
        -------
        Subspace
            The subspace object

        Raises
        ------
        KeyError
            If subspace not found
        """
        if name not in self._subspaces:
            raise KeyError(f"Subspace '{name}' not registered. "
                          f"Available: {list(self._subspaces.keys())}")
        return self._subspaces[name]

    @property
    def subspaces(self) -> Dict[str, Subspace]:
        """Dictionary of all registered subspaces."""
        return self._subspaces.copy()

    @abstractmethod
    def noise_operators(self) -> Dict[str, np.ndarray]:
        """
        Return the noise operators for this system.

        Returns
        -------
        dict
            Mapping from noise type name to operator matrix.
            Common types: 'dephasing', 'amplitude', 'leakage'
        """
        pass

    @abstractmethod
    def default_measurement_operator(self) -> np.ndarray:
        """
        Return the default measurement operator for this system.

        Returns
        -------
        np.ndarray
            Measurement operator (Hermitian matrix)
        """
        pass

    def projector(self, level: int) -> np.ndarray:
        """Return projector onto a single level."""
        return self._hilbert.projector(level)

    def population_operator(self, level: int) -> np.ndarray:
        """
        Return population measurement operator for a level.

        This is the same as the projector |level><level|.

        Parameters
        ----------
        level : int
            Level index

        Returns
        -------
        np.ndarray
            Projector onto the level
        """
        return self.projector(level)

    def prepare_state(self, state_spec) -> np.ndarray:
        """
        Prepare an initial state.

        Parameters
        ----------
        state_spec : int, np.ndarray, or str
            If int: prepare basis state |state_spec>
            If np.ndarray: use directly (normalized)
            If str: interpret as named state (system-dependent)

        Returns
        -------
        np.ndarray
            Normalized state vector
        """
        if isinstance(state_spec, int):
            return self._hilbert.basis_state(state_spec)
        elif isinstance(state_spec, np.ndarray):
            state = np.asarray(state_spec, dtype=complex)
            return state / np.linalg.norm(state)
        elif isinstance(state_spec, str):
            return self._prepare_named_state(state_spec)
        else:
            raise TypeError(f"Unknown state specification type: {type(state_spec)}")

    def _prepare_named_state(self, name: str) -> np.ndarray:
        """
        Prepare a named state. Override in subclasses.

        Parameters
        ----------
        name : str
            State name (e.g., 'ground', 'superposition')

        Returns
        -------
        np.ndarray
            State vector
        """
        raise NotImplementedError(f"Named state '{name}' not defined for {type(self)}")

    def expectation_value(self, operator: np.ndarray, state: np.ndarray) -> complex:
        """
        Compute expectation value <state|operator|state>.

        Parameters
        ----------
        operator : np.ndarray
            Operator matrix
        state : np.ndarray
            State vector

        Returns
        -------
        complex
            Expectation value
        """
        return np.vdot(state, operator @ state)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"
