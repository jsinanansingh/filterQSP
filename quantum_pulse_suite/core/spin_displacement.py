"""
Spin-displacement pulses for qudit quantum signal processing.

Implements the multi-tone spin-displacement pulses from QSP Clock Draft
Section 5.2. When Rabi frequencies satisfy Omega_k = Omega * sqrt((k+1)(d-k-1)),
the Hamiltonian reduces to Omega * Jx (spin-j x-operator with j=(d-1)/2).

The key objects are:
- Spin-j angular momentum operators Jx, Jy, Jz
- H_rot(phases, detunings): the rotating-frame multi-tone Hamiltonian
- D(phases, theta): spin-displacement pulse unitary
- S(phases): virtual SNAP gate (diagonal phase gate)

When phases and detunings are zero, D(0, theta) = exp(-i*theta*Jx),
which is a pure SU(2) rotation in the spin-j representation.
"""

import numpy as np
from scipy.linalg import expm


def spin_j_operators(d):
    """
    Compute spin-j angular momentum operators for a d-level system.

    j = (d-1)/2, so d=2 gives spin-1/2, d=3 gives spin-1, etc.

    Parameters
    ----------
    d : int
        Hilbert space dimension.

    Returns
    -------
    Jx, Jy, Jz : np.ndarray
        d x d Hermitian matrices satisfying [Jx, Jy] = i*Jz (cyclic).
        Jz = diag(j, j-1, ..., -j) in the standard |j,m> basis with
        m decreasing (m=j is index 0, m=-j is index d-1).
    """
    j = (d - 1) / 2.0
    m_values = np.array([j - k for k in range(d)])  # j, j-1, ..., -j

    # Jz is diagonal
    Jz = np.diag(m_values).astype(complex)

    # J+ (raising operator): J+|j,m> = sqrt(j(j+1) - m(m+1)) |j,m+1>
    # In our indexing, m+1 corresponds to index k-1
    Jp = np.zeros((d, d), dtype=complex)
    for k in range(1, d):
        m = m_values[k]  # m value at index k
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - m * (m + 1))

    Jm = Jp.conj().T

    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0j)

    return Jx, Jy, Jz


def spin_displacement_hamiltonian(d, phases=None, omega=1.0, detunings=None):
    r"""
    Build the rotating-frame multi-tone Hamiltonian H_rot.

    From Eq. (eq:H_rot) of the QSP Clock Draft:

        H_rot = sum_k Omega_k (e^{i*phi_k} |k><k+1| + h.c.) + detuning terms

    where Omega_k = omega * sqrt((k+1)(d-k-1)). When phases=0 and detunings=0,
    this reduces to omega * Jx.

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    phases : array_like of length d-1, optional
        Phases phi_k for each tone. Default: all zeros (gives Jx).
    omega : float
        Base Rabi frequency.
    detunings : array_like of length d-1, optional
        Detunings delta_k for each transition. Default: all zeros.

    Returns
    -------
    H : np.ndarray, shape (d, d)
        Hermitian Hamiltonian matrix.
    """
    if phases is None:
        phases = np.zeros(d - 1)
    else:
        phases = np.asarray(phases, dtype=float)
        if len(phases) != d - 1:
            raise ValueError(f"phases must have length d-1={d-1}, got {len(phases)}")

    if detunings is None:
        detunings = np.zeros(d - 1)
    else:
        detunings = np.asarray(detunings, dtype=float)
        if len(detunings) != d - 1:
            raise ValueError(f"detunings must have length d-1={d-1}, got {len(detunings)}")

    j = (d - 1) / 2.0
    m_values = np.array([j - k for k in range(d)])

    H = np.zeros((d, d), dtype=complex)

    # Off-diagonal: coupling between |k> and |k+1>
    # Full Rabi rate: Omega_k = omega * sqrt((k+1)(d-k-1))
    # Hamiltonian coupling: (Omega_k / 2) * (e^{i*phi_k} |k><k+1| + h.c.)
    # The factor of 1/2 ensures H_rot(phases=0) = omega * Jx, where Jx
    # has matrix elements (Jx)_{k,k+1} = (1/2)*sqrt((k+1)(d-k-1)).
    for k in range(d - 1):
        coupling_k = omega * np.sqrt((k + 1) * (d - k - 1)) / 2.0
        H[k, k + 1] = coupling_k * np.exp(1j * phases[k])
        H[k + 1, k] = coupling_k * np.exp(-1j * phases[k])

    # Diagonal: cumulative detunings
    # delta_k shifts level |k+1> relative to |k>
    # So level |k> has shift sum_{l=0}^{k-1} delta_l
    cumulative_detuning = 0.0
    for k in range(d - 1):
        cumulative_detuning += detunings[k]
        H[k + 1, k + 1] += cumulative_detuning

    return H


def spin_displacement_pulse(d, phases=None, theta=np.pi, omega=1.0,
                            detunings=None):
    """
    Compute the spin-displacement pulse unitary D(phases, theta).

    D(phases, theta) = expm(-i * theta * H_rot(phases) / omega)

    When phases=0 and detunings=0: D(0, theta) = expm(-i * theta * Jx).

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    phases : array_like, optional
        Phases for each tone.
    theta : float
        Rotation angle (pulse area). Default: pi.
    omega : float
        Base Rabi frequency.
    detunings : array_like, optional
        Detunings for each transition.

    Returns
    -------
    D : np.ndarray, shape (d, d)
        Unitary pulse matrix.
    """
    H = spin_displacement_hamiltonian(d, phases, omega, detunings)
    return expm(-1j * theta * H / omega)


def snap_gate(d, phases):
    """
    Compute a virtual SNAP gate (diagonal phase gate).

    S(phases) = diag(exp(i*phi_0), exp(i*phi_1), ..., exp(i*phi_{d-1}))

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    phases : array_like of length d
        Phase for each level.

    Returns
    -------
    S : np.ndarray, shape (d, d)
        Diagonal unitary matrix.
    """
    phases = np.asarray(phases, dtype=float)
    if len(phases) != d:
        raise ValueError(f"phases must have length d={d}, got {len(phases)}")
    return np.diag(np.exp(1j * phases))


def su2_scaling_factor(d):
    """
    Compute the noise susceptibility scaling factor for SU(2)-embedded qudits.

    When a d-level system is driven by pure Jx/Jy/Jz rotations (SU(2) subgroup)
    with Jz dephasing noise, the total noise susceptibility scales relative to
    the qubit case by:

        ratio = d(d^2 - 1) / 6

    This equals 2*Tr(Jz^2) for j=(d-1)/2 divided by 2*Tr((sigma_z/2)^2)=1.

    Parameters
    ----------
    d : int
        Hilbert space dimension.

    Returns
    -------
    float
        Scaling factor. d=2 gives 1, d=3 gives 4, d=4 gives 10, etc.
    """
    return d * (d**2 - 1) / 6
