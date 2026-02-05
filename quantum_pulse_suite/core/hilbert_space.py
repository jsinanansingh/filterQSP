"""
Hilbert space and subspace definitions for multi-level quantum systems.

This module provides dimension-agnostic representations of quantum systems,
enabling pulse sequences and filter functions on subspaces of larger systems.
"""

from typing import Tuple, List, Optional, Union
import numpy as np
from abc import ABC, abstractmethod


class HilbertSpace:
    """
    Represents a d-dimensional Hilbert space.

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space
    level_labels : list of str, optional
        Human-readable labels for each level (e.g., ['g', 'e', 'm'])

    Examples
    --------
    >>> space = HilbertSpace(3, level_labels=['g', 'e', 'm'])
    >>> space.dim
    3
    >>> space.identity
    array([[1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j]])
    """

    def __init__(self, dim: int, level_labels: Optional[List[str]] = None):
        if dim < 2:
            raise ValueError("Hilbert space dimension must be at least 2")

        self._dim = dim
        self._identity = np.eye(dim, dtype=complex)

        if level_labels is None:
            self._level_labels = [str(i) for i in range(dim)]
        else:
            if len(level_labels) != dim:
                raise ValueError(f"Expected {dim} labels, got {len(level_labels)}")
            self._level_labels = list(level_labels)

    @property
    def dim(self) -> int:
        """Dimension of the Hilbert space."""
        return self._dim

    @property
    def identity(self) -> np.ndarray:
        """Identity operator on this space."""
        return self._identity.copy()

    @property
    def level_labels(self) -> List[str]:
        """Labels for each level."""
        return self._level_labels.copy()

    def basis_state(self, level: int) -> np.ndarray:
        """
        Return the basis state |level>.

        Parameters
        ----------
        level : int
            Level index (0-indexed)

        Returns
        -------
        np.ndarray
            Column vector representing |level>
        """
        if not 0 <= level < self._dim:
            raise ValueError(f"Level {level} out of range [0, {self._dim})")
        state = np.zeros(self._dim, dtype=complex)
        state[level] = 1.0
        return state

    def projector(self, level: int) -> np.ndarray:
        """
        Return the projector |level><level|.

        Parameters
        ----------
        level : int
            Level index

        Returns
        -------
        np.ndarray
            Projector matrix
        """
        state = self.basis_state(level)
        return np.outer(state, state.conj())

    def transition_operator(self, i: int, j: int) -> np.ndarray:
        """
        Return the transition operator |i><j|.

        Parameters
        ----------
        i, j : int
            Level indices

        Returns
        -------
        np.ndarray
            Transition matrix with 1 at position (i, j)
        """
        if not (0 <= i < self._dim and 0 <= j < self._dim):
            raise ValueError(f"Levels {i}, {j} out of range [0, {self._dim})")
        op = np.zeros((self._dim, self._dim), dtype=complex)
        op[i, j] = 1.0
        return op

    def __repr__(self) -> str:
        return f"HilbertSpace(dim={self._dim}, labels={self._level_labels})"


class Subspace:
    """
    A 2-level subspace within a larger Hilbert space.

    This class handles embedding of 2x2 operators into the full space
    and projection of full-space states/operators onto the subspace.

    Parameters
    ----------
    parent : HilbertSpace
        The full Hilbert space
    levels : tuple of int
        The two levels defining this subspace (ordered as |0>, |1> of subspace)
    label : str, optional
        Human-readable label for this subspace (e.g., 'optical', 'metastable')

    Examples
    --------
    >>> space = HilbertSpace(3, level_labels=['g', 'e', 'm'])
    >>> optical = Subspace(space, levels=(0, 1), label='optical')
    >>> optical.embed_operator(SIGMA_X)  # Embed Pauli X into 3x3 space
    """

    def __init__(self, parent: HilbertSpace, levels: Tuple[int, int],
                 label: Optional[str] = None):
        if len(levels) != 2:
            raise ValueError("Subspace must be defined by exactly 2 levels")
        if levels[0] == levels[1]:
            raise ValueError("Subspace levels must be distinct")
        if not all(0 <= l < parent.dim for l in levels):
            raise ValueError(f"Levels {levels} out of range [0, {parent.dim})")

        self._parent = parent
        self._levels = tuple(levels)
        self._label = label or f"subspace_{levels[0]}_{levels[1]}"

        # Build embedding and projection matrices
        self._build_embedding_matrices()

    def _build_embedding_matrices(self):
        """Build the matrices for embedding and projection."""
        d = self._parent.dim
        i, j = self._levels

        # Embedding matrix: 2 -> d
        # Maps |0>_sub to |i>_full and |1>_sub to |j>_full
        self._embed_mat = np.zeros((d, 2), dtype=complex)
        self._embed_mat[i, 0] = 1.0
        self._embed_mat[j, 1] = 1.0

        # Projection matrix: d -> 2
        self._project_mat = self._embed_mat.conj().T

    @property
    def parent(self) -> HilbertSpace:
        """The parent Hilbert space."""
        return self._parent

    @property
    def levels(self) -> Tuple[int, int]:
        """The two levels defining this subspace."""
        return self._levels

    @property
    def label(self) -> str:
        """Label for this subspace."""
        return self._label

    @property
    def dim(self) -> int:
        """Dimension of the subspace (always 2)."""
        return 2

    def embed_state(self, state_2d: np.ndarray) -> np.ndarray:
        """
        Embed a 2D state vector into the full Hilbert space.

        Parameters
        ----------
        state_2d : np.ndarray
            2-component state vector in subspace

        Returns
        -------
        np.ndarray
            d-component state vector in full space
        """
        state_2d = np.asarray(state_2d, dtype=complex)
        if state_2d.shape != (2,):
            raise ValueError(f"Expected 2D state, got shape {state_2d.shape}")
        return self._embed_mat @ state_2d

    def project_state(self, state_full: np.ndarray) -> np.ndarray:
        """
        Project a full-space state onto this subspace.

        Note: This does NOT normalize. The projected state may have
        norm < 1 if there is population outside the subspace.

        Parameters
        ----------
        state_full : np.ndarray
            d-component state vector

        Returns
        -------
        np.ndarray
            2-component state vector
        """
        state_full = np.asarray(state_full, dtype=complex)
        if state_full.shape != (self._parent.dim,):
            raise ValueError(f"Expected {self._parent.dim}D state")
        return self._project_mat @ state_full

    def embed_operator(self, op_2x2: np.ndarray) -> np.ndarray:
        """
        Embed a 2x2 operator into the full Hilbert space.

        The embedded operator acts as identity on levels outside the subspace.

        Parameters
        ----------
        op_2x2 : np.ndarray
            2x2 operator matrix

        Returns
        -------
        np.ndarray
            d x d operator matrix
        """
        op_2x2 = np.asarray(op_2x2, dtype=complex)
        if op_2x2.shape != (2, 2):
            raise ValueError(f"Expected 2x2 operator, got shape {op_2x2.shape}")

        # Start with identity
        full_op = self._parent.identity.copy()

        # Replace the 2x2 block
        i, j = self._levels
        full_op[i, i] = op_2x2[0, 0]
        full_op[i, j] = op_2x2[0, 1]
        full_op[j, i] = op_2x2[1, 0]
        full_op[j, j] = op_2x2[1, 1]

        return full_op

    def embed_operator_zero_outside(self, op_2x2: np.ndarray) -> np.ndarray:
        """
        Embed a 2x2 operator with zeros outside the subspace.

        Unlike embed_operator, this does NOT act as identity elsewhere.
        Useful for projectors and transition operators.

        Parameters
        ----------
        op_2x2 : np.ndarray
            2x2 operator matrix

        Returns
        -------
        np.ndarray
            d x d operator matrix (zeros outside subspace)
        """
        op_2x2 = np.asarray(op_2x2, dtype=complex)
        if op_2x2.shape != (2, 2):
            raise ValueError(f"Expected 2x2 operator, got shape {op_2x2.shape}")

        return self._embed_mat @ op_2x2 @ self._project_mat

    def project_operator(self, op_full: np.ndarray) -> np.ndarray:
        """
        Project a full-space operator onto this subspace.

        Parameters
        ----------
        op_full : np.ndarray
            d x d operator matrix

        Returns
        -------
        np.ndarray
            2x2 operator matrix
        """
        op_full = np.asarray(op_full, dtype=complex)
        d = self._parent.dim
        if op_full.shape != (d, d):
            raise ValueError(f"Expected {d}x{d} operator")

        i, j = self._levels
        return np.array([
            [op_full[i, i], op_full[i, j]],
            [op_full[j, i], op_full[j, j]]
        ], dtype=complex)

    def subspace_pauli_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return Pauli matrices embedded in the full space for this subspace.

        Returns
        -------
        tuple
            (sigma_x, sigma_y, sigma_z) as d x d matrices
        """
        from .pulse_sequence import SIGMA_X, SIGMA_Y, SIGMA_Z
        return (
            self.embed_operator_zero_outside(SIGMA_X),
            self.embed_operator_zero_outside(SIGMA_Y),
            self.embed_operator_zero_outside(SIGMA_Z)
        )

    def __repr__(self) -> str:
        parent_labels = self._parent.level_labels
        l0, l1 = self._levels
        return (f"Subspace('{self._label}': |{parent_labels[l0]}> <-> "
                f"|{parent_labels[l1]}>)")


class LiouvilleSpace:
    """
    Liouville (operator) space for a quantum system.

    For a d-dimensional Hilbert space, the Liouville space is d^2-dimensional.
    This class provides tools for working with density matrices as vectors
    and superoperators.

    Parameters
    ----------
    hilbert_space : HilbertSpace
        The underlying Hilbert space

    Notes
    -----
    Vectorization convention: rho -> |rho>> uses column-major (Fortran) ordering.
    rho_vec[i*d + j] = rho[j, i]
    """

    def __init__(self, hilbert_space: HilbertSpace):
        self._hilbert = hilbert_space
        self._dim = hilbert_space.dim ** 2

    @property
    def hilbert_space(self) -> HilbertSpace:
        """The underlying Hilbert space."""
        return self._hilbert

    @property
    def dim(self) -> int:
        """Dimension of Liouville space (d^2)."""
        return self._dim

    @property
    def hilbert_dim(self) -> int:
        """Dimension of the Hilbert space (d)."""
        return self._hilbert.dim

    def vectorize(self, rho: np.ndarray) -> np.ndarray:
        """
        Convert density matrix to vector in Liouville space.

        Parameters
        ----------
        rho : np.ndarray
            d x d density matrix

        Returns
        -------
        np.ndarray
            d^2 component vector
        """
        d = self._hilbert.dim
        rho = np.asarray(rho, dtype=complex)
        if rho.shape != (d, d):
            raise ValueError(f"Expected {d}x{d} matrix")
        return rho.flatten(order='F')

    def unvectorize(self, rho_vec: np.ndarray) -> np.ndarray:
        """
        Convert Liouville vector back to density matrix.

        Parameters
        ----------
        rho_vec : np.ndarray
            d^2 component vector

        Returns
        -------
        np.ndarray
            d x d density matrix
        """
        d = self._hilbert.dim
        rho_vec = np.asarray(rho_vec, dtype=complex)
        if rho_vec.shape != (d*d,):
            raise ValueError(f"Expected {d*d} component vector")
        return rho_vec.reshape((d, d), order='F')

    def superoperator(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Build superoperator L such that L|rho>> = |left @ rho @ right^dag>>.

        Parameters
        ----------
        left : np.ndarray
            Left operator
        right : np.ndarray
            Right operator (will be conjugated)

        Returns
        -------
        np.ndarray
            d^2 x d^2 superoperator matrix
        """
        # L = left ⊗ right^*
        return np.kron(left, right.conj())

    def commutator_superoperator(self, H: np.ndarray) -> np.ndarray:
        """
        Build superoperator for commutator: L|rho>> = |[H, rho]>>.

        Parameters
        ----------
        H : np.ndarray
            Hamiltonian

        Returns
        -------
        np.ndarray
            d^2 x d^2 superoperator
        """
        d = self._hilbert.dim
        I = self._hilbert.identity
        # [H, rho] = H @ rho - rho @ H
        # As superoperator: H ⊗ I - I ⊗ H^T
        return np.kron(H, I) - np.kron(I, H.T)

    def basis_operators(self) -> List[np.ndarray]:
        """
        Return an orthonormal basis for the operator space.

        Uses the generalized Gell-Mann matrices plus identity/sqrt(d).

        Returns
        -------
        list of np.ndarray
            d^2 orthonormal operators (Frobenius inner product)
        """
        from .operators import generalized_gell_mann_matrices

        d = self._hilbert.dim
        basis = []

        # Identity normalized
        basis.append(self._hilbert.identity / np.sqrt(d))

        # Generalized Gell-Mann matrices (already normalized)
        gell_mann = generalized_gell_mann_matrices(d)
        basis.extend(gell_mann)

        return basis
