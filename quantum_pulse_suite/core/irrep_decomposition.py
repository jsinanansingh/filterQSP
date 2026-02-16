"""
Irreducible representation decomposition of qudit filter functions.

Decomposes the matrix-valued filter function F(w) into angular momentum
irrep components. In the spin-j representation (j=(d-1)/2), the Liouville
space j ⊗ j* decomposes as:

    j ⊗ j* = 0 ⊕ 1 ⊕ ... ⊕ 2j

The total angular momentum super-operator L_i(A) = [J_i, A] has Casimir
L^2 with eigenvalues L(L+1) for L=0,...,2j. The L=0 sector is spanned by
the identity (trace part). The L=1 sector corresponds to the vector
(dipole) operators, L=2 to the quadrupole, etc.

For Jz dephasing noise at first order, only the L=1 sector contributes,
since Jz ∈ L=1. This provides a physical decomposition of the noise
susceptibility by multipole order.

Three decomposition methods are provided:
1. Angular momentum irrep decomposition (by total L)
2. Transition-resolved decomposition (by |j><k| matrix element)
3. GGM component decomposition (by generalized Gell-Mann basis element)
"""

import numpy as np
from .spin_displacement import spin_j_operators
from .operators import generalized_gell_mann_matrices
from .fft_filter_function import (
    noise_susceptibility_from_matrix,
    bloch_components_from_matrix,
)


def _superoperator_commutator(J, d):
    """Build the d^2 x d^2 super-operator for A -> [J, A].

    Vectorization convention: vec(A) with A[i,j] -> index i*d + j.
    Then [J, A] = J⊗I - I⊗J^T in the vec basis.

    Parameters
    ----------
    J : np.ndarray, shape (d, d)
        Operator to commute with.
    d : int
        Hilbert space dimension.

    Returns
    -------
    np.ndarray, shape (d^2, d^2)
        Super-operator matrix.
    """
    Id = np.eye(d, dtype=complex)
    return np.kron(J, Id) - np.kron(Id, J.T)


def angular_momentum_irrep_projectors(d):
    """Compute projectors for angular momentum irreps in Liouville space.

    For a d-level system with j=(d-1)/2, the Liouville space decomposes as
    j ⊗ j* = 0 ⊕ 1 ⊕ ... ⊕ 2j. This function returns the projector onto
    each irrep sector.

    The projectors are computed by diagonalizing the Casimir operator
    L^2 = L_x^2 + L_y^2 + L_z^2 where L_i(A) = [J_i, A].

    Parameters
    ----------
    d : int
        Hilbert space dimension.

    Returns
    -------
    projectors : dict
        {L: P_L} where P_L is a (d^2, d^2) projector matrix acting on
        vectorized d×d operators. L ranges over 0, 1, ..., 2j.
    """
    Jx, Jy, Jz = spin_j_operators(d)

    Lx = _superoperator_commutator(Jx, d)
    Ly = _superoperator_commutator(Jy, d)
    Lz = _superoperator_commutator(Jz, d)

    L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz

    # L^2 is real symmetric (the commutator super-operators for Hermitian
    # generators produce a real Casimir). Force real to avoid numerical
    # issues with complex eigh on degenerate eigenspaces.
    L2_real = np.real((L2 + L2.conj().T) / 2)

    # Diagonalize L^2
    eigvals, eigvecs = np.linalg.eigh(L2_real)

    # Group eigenvalues by L(L+1) quantum number
    j = (d - 1) / 2.0
    max_L = int(2 * j)

    projectors = {}
    for L in range(max_L + 1):
        target = L * (L + 1)
        # Find eigenvectors with eigenvalue ≈ L(L+1)
        mask = np.abs(eigvals - target) < 0.5  # eigenvalues are integers
        if np.any(mask):
            V = eigvecs[:, mask]  # columns are the eigenvectors
            P_L = V @ V.conj().T
            projectors[L] = P_L

    return projectors


def irrep_resolved_filter_function(F_matrix, d):
    """Decompose the matrix filter function by angular momentum irreps.

    Projects F(w) onto each irrep sector L=0,...,2j and computes the
    noise susceptibility contribution from each sector:

        F_L(w) = 2 * Tr(P_L[F(w)] @ P_L[F(w)]†)

    where P_L[F] reshapes F to a d²-vector, applies the projector,
    and reshapes back.

    Parameters
    ----------
    F_matrix : np.ndarray, shape (n_freq, d, d)
        Matrix-valued filter function from fft_filter_function.
    d : int
        Hilbert space dimension.

    Returns
    -------
    irrep_susceptibilities : dict
        {L: F_L(w)} where F_L is (n_freq,) array giving the noise
        susceptibility from irrep L at each frequency.
    """
    projectors = angular_momentum_irrep_projectors(d)
    n_freq = F_matrix.shape[0]

    # Vectorize F_matrix: (n_freq, d, d) -> (n_freq, d^2)
    F_vec = F_matrix.reshape(n_freq, d * d)

    irrep_susceptibilities = {}
    for L, P_L in projectors.items():
        # Project: F_L_vec = P_L @ F_vec^T -> (d^2, n_freq)
        F_L_vec = (P_L @ F_vec.T).T  # (n_freq, d^2)
        F_L_mat = F_L_vec.reshape(n_freq, d, d)
        irrep_susceptibilities[L] = noise_susceptibility_from_matrix(F_L_mat)

    return irrep_susceptibilities


def transition_resolved_filter_function(F_matrix, d):
    """Decompose the matrix filter function by transition pairs.

    For each pair (j,k) with j<k, computes the contribution from
    the off-diagonal matrix elements F_{jk}(w). Also returns the
    diagonal contribution.

    The off-diagonal contribution from (j,k) is:
        F_{jk}(w) = 2 * (|F_{jk}|^2 + |F_{kj}|^2)

    The diagonal contribution is:
        F_diag(w) = 2 * sum_j |F_{jj}|^2

    These sum to the total susceptibility.

    Parameters
    ----------
    F_matrix : np.ndarray, shape (n_freq, d, d)
        Matrix-valued filter function.
    d : int
        Hilbert space dimension.

    Returns
    -------
    transition_susceptibilities : dict
        {(j,k): F_{jk}(w)} for j<k (off-diagonal transitions)
        plus {('diag',): F_diag(w)} for the diagonal part.
        Each value is (n_freq,) array.
    """
    result = {}

    # Diagonal contribution
    diag_sum = np.zeros(F_matrix.shape[0])
    for j in range(d):
        diag_sum += 2.0 * np.abs(F_matrix[:, j, j])**2
    result[('diag',)] = diag_sum

    # Off-diagonal contributions (j < k)
    for j in range(d):
        for k in range(j + 1, d):
            f_jk = 2.0 * (np.abs(F_matrix[:, j, k])**2 +
                           np.abs(F_matrix[:, k, j])**2)
            result[(j, k)] = f_jk

    return result


def ggm_component_filter_functions(F_matrix, d):
    """Compute |R_a(w)|^2 for each GGM component.

    Wraps bloch_components_from_matrix with the generalized Gell-Mann
    basis for dimension d.

    Parameters
    ----------
    F_matrix : np.ndarray, shape (n_freq, d, d)
        Matrix-valued filter function.
    d : int
        Hilbert space dimension.

    Returns
    -------
    component_susceptibilities : np.ndarray, shape (d^2-1, n_freq)
        |R_a(w)|^2 for each GGM component a=0,...,d^2-2.
    ggm_labels : list of str
        Labels for each GGM component (e.g., 'sym_01', 'asym_01', 'diag_1').
    """
    ggm = generalized_gell_mann_matrices(d)
    components = bloch_components_from_matrix(F_matrix, ggm)

    n_ggm = len(ggm)
    n_freq = F_matrix.shape[0]
    susceptibilities = np.zeros((n_ggm, n_freq))
    for a, c in enumerate(components):
        susceptibilities[a] = np.abs(c)**2

    # Generate labels
    labels = []
    idx = 0
    # Symmetric off-diagonal
    for j in range(d):
        for k in range(j + 1, d):
            labels.append(f'sym_{j}{k}')
            idx += 1
    # Antisymmetric off-diagonal
    for j in range(d):
        for k in range(j + 1, d):
            labels.append(f'asym_{j}{k}')
            idx += 1
    # Diagonal
    for l in range(1, d):
        labels.append(f'diag_{l}')
        idx += 1

    return susceptibilities, labels


def irrep_multiplicities(d):
    """Return the expected multiplicity (dimension) of each irrep sector.

    For j ⊗ j* = 0 ⊕ 1 ⊕ ... ⊕ 2j, irrep L has dimension 2L+1.
    As a check: sum of (2L+1) for L=0,...,2j = d^2.

    Parameters
    ----------
    d : int
        Hilbert space dimension.

    Returns
    -------
    dict
        {L: 2L+1} for L=0,...,2j.
    """
    j = (d - 1) / 2.0
    max_L = int(2 * j)
    return {L: 2 * L + 1 for L in range(max_L + 1)}
