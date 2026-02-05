"""
Generalized operator algebras for multi-level quantum systems.

This module provides:
- Pauli matrices (SU(2) generators) for qubits
- Gell-Mann matrices (SU(3) generators) for qutrits
- Generalized Gell-Mann matrices (SU(d) generators) for qudits

All generators are traceless Hermitian matrices satisfying:
    Tr(T_a T_b) = (1/2) delta_ab

References
----------
- Bertlmann & Krammer, "Bloch vectors for qudits", J. Phys. A 41, 235303 (2008)
"""

from typing import List, Tuple
import numpy as np


# =============================================================================
# Pauli Matrices (d=2)
# =============================================================================

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

PAULI_MATRICES = [SIGMA_X, SIGMA_Y, SIGMA_Z]


def pauli_matrices() -> List[np.ndarray]:
    """Return the three Pauli matrices [sigma_x, sigma_y, sigma_z]."""
    return [m.copy() for m in PAULI_MATRICES]


# =============================================================================
# Gell-Mann Matrices (d=3)
# =============================================================================

def gell_mann_matrices() -> List[np.ndarray]:
    """
    Return the 8 Gell-Mann matrices for SU(3).

    Normalized such that Tr(lambda_a lambda_b) = 2 delta_ab.

    Returns
    -------
    list of np.ndarray
        8 Gell-Mann matrices [lambda_1, ..., lambda_8]
    """
    # Symmetric off-diagonal (like sigma_x)
    l1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    l4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    l6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)

    # Antisymmetric off-diagonal (like sigma_y)
    l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
    l5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
    l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)

    # Diagonal (like sigma_z)
    l3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
    l8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)

    return [l1, l2, l3, l4, l5, l6, l7, l8]


# =============================================================================
# Generalized Gell-Mann Matrices (d arbitrary)
# =============================================================================

def generalized_gell_mann_matrices(d: int) -> List[np.ndarray]:
    """
    Return the d^2 - 1 generalized Gell-Mann matrices for SU(d).

    These are traceless Hermitian matrices forming a basis for su(d).
    Normalized such that Tr(T_a T_b) = 2 delta_ab.

    The generators consist of:
    - d(d-1)/2 symmetric matrices (generalized sigma_x)
    - d(d-1)/2 antisymmetric matrices (generalized sigma_y)
    - d-1 diagonal matrices (generalized sigma_z)

    Total: d(d-1)/2 + d(d-1)/2 + (d-1) = d^2 - 1

    Parameters
    ----------
    d : int
        Dimension of the Hilbert space

    Returns
    -------
    list of np.ndarray
        d^2 - 1 generalized Gell-Mann matrices
    """
    if d < 2:
        raise ValueError("Dimension must be at least 2")

    if d == 2:
        return pauli_matrices()

    if d == 3:
        return gell_mann_matrices()

    matrices = []

    # Symmetric off-diagonal matrices (like sigma_x)
    # For each pair (j, k) with j < k, create matrix with 1 at (j,k) and (k,j)
    for j in range(d):
        for k in range(j + 1, d):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = 1
            mat[k, j] = 1
            matrices.append(mat)

    # Antisymmetric off-diagonal matrices (like sigma_y)
    # For each pair (j, k) with j < k, create matrix with -i at (j,k) and i at (k,j)
    for j in range(d):
        for k in range(j + 1, d):
            mat = np.zeros((d, d), dtype=complex)
            mat[j, k] = -1j
            mat[k, j] = 1j
            matrices.append(mat)

    # Diagonal matrices (like sigma_z)
    # For l = 1, ..., d-1: sqrt(2/(l(l+1))) * diag(1, ..., 1, -l, 0, ..., 0)
    for l in range(1, d):
        mat = np.zeros((d, d), dtype=complex)
        norm = np.sqrt(2 / (l * (l + 1)))
        for j in range(l):
            mat[j, j] = norm
        mat[l, l] = -l * norm
        matrices.append(mat)

    return matrices


def su_d_structure_constants(d: int) -> np.ndarray:
    """
    Compute the structure constants f_abc for SU(d).

    The generators satisfy [T_a, T_b] = i f_abc T_c (Einstein summation).

    Parameters
    ----------
    d : int
        Dimension

    Returns
    -------
    np.ndarray
        Structure constants f[a, b, c] with shape (d^2-1, d^2-1, d^2-1)
    """
    generators = generalized_gell_mann_matrices(d)
    n = len(generators)  # d^2 - 1

    f = np.zeros((n, n, n), dtype=complex)

    for a in range(n):
        for b in range(n):
            # Compute commutator [T_a, T_b]
            comm = generators[a] @ generators[b] - generators[b] @ generators[a]

            # Project onto each generator: f_abc = -i Tr([T_a, T_b] T_c) / Tr(T_c^2)
            for c in range(n):
                # Tr(T_c^2) = 2 for our normalization
                f[a, b, c] = -1j * np.trace(comm @ generators[c]) / 2

    return np.real(f)  # Structure constants are real for SU(d)


# =============================================================================
# Operator Utilities
# =============================================================================

def rotation_operator(generators: List[np.ndarray], axis: np.ndarray,
                      angle: float) -> np.ndarray:
    """
    Compute rotation operator exp(-i * (axis . generators) * angle / 2).

    Parameters
    ----------
    generators : list of np.ndarray
        The generator matrices (e.g., Pauli matrices)
    axis : np.ndarray
        Rotation axis (will be normalized)
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray
        Rotation operator (unitary matrix)
    """
    from scipy.linalg import expm

    axis = np.asarray(axis, dtype=float)
    if len(axis) != len(generators):
        raise ValueError(f"Axis dimension {len(axis)} != number of generators {len(generators)}")

    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        # Return identity for zero rotation
        d = generators[0].shape[0]
        return np.eye(d, dtype=complex)

    axis = axis / norm

    # Build Hamiltonian H = axis . generators
    H = sum(a * g for a, g in zip(axis, generators))

    return expm(-1j * H * angle / 2)


def transition_matrix(d: int, i: int, j: int) -> np.ndarray:
    """
    Return the transition matrix |i><j| for a d-level system.

    Parameters
    ----------
    d : int
        Dimension
    i, j : int
        Level indices

    Returns
    -------
    np.ndarray
        d x d matrix with 1 at position (i, j)
    """
    mat = np.zeros((d, d), dtype=complex)
    mat[i, j] = 1.0
    return mat


def lowering_operator(d: int, i: int, j: int) -> np.ndarray:
    """
    Return lowering operator for transition i -> j (like sigma_minus).

    Parameters
    ----------
    d : int
        Dimension
    i, j : int
        Upper and lower level indices

    Returns
    -------
    np.ndarray
        |j><i|
    """
    return transition_matrix(d, j, i)


def raising_operator(d: int, i: int, j: int) -> np.ndarray:
    """
    Return raising operator for transition j -> i (like sigma_plus).

    Parameters
    ----------
    d : int
        Dimension
    i, j : int
        Upper and lower level indices

    Returns
    -------
    np.ndarray
        |i><j|
    """
    return transition_matrix(d, i, j)


def x_operator(d: int, i: int, j: int) -> np.ndarray:
    """
    Return sigma_x-like operator for levels i and j.

    sigma_x^{ij} = |i><j| + |j><i|

    Parameters
    ----------
    d : int
        Dimension
    i, j : int
        Level indices

    Returns
    -------
    np.ndarray
        d x d Hermitian matrix
    """
    return transition_matrix(d, i, j) + transition_matrix(d, j, i)


def y_operator(d: int, i: int, j: int) -> np.ndarray:
    """
    Return sigma_y-like operator for levels i and j.

    sigma_y^{ij} = -i|i><j| + i|j><i|

    Parameters
    ----------
    d : int
        Dimension
    i, j : int
        Level indices

    Returns
    -------
    np.ndarray
        d x d Hermitian matrix
    """
    return -1j * transition_matrix(d, i, j) + 1j * transition_matrix(d, j, i)


def z_operator(d: int, i: int, j: int) -> np.ndarray:
    """
    Return sigma_z-like operator for levels i and j.

    sigma_z^{ij} = |i><i| - |j><j|

    Parameters
    ----------
    d : int
        Dimension
    i, j : int
        Level indices

    Returns
    -------
    np.ndarray
        d x d diagonal Hermitian matrix
    """
    mat = np.zeros((d, d), dtype=complex)
    mat[i, i] = 1
    mat[j, j] = -1
    return mat


def subspace_pauli(d: int, levels: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return Pauli-like matrices for a 2-level subspace of a d-level system.

    Parameters
    ----------
    d : int
        Total system dimension
    levels : tuple of int
        The two levels (i, j) defining the subspace

    Returns
    -------
    tuple of np.ndarray
        (sigma_x, sigma_y, sigma_z) embedded in d x d space
    """
    i, j = levels
    return (
        x_operator(d, i, j),
        y_operator(d, i, j),
        z_operator(d, i, j)
    )
