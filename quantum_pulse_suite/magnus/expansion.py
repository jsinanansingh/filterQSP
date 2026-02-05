"""
Magnus expansion computation for time-ordered evolution.

This module provides tools for computing the Magnus expansion terms
Omega_1, Omega_2, etc., which describe the effective Hamiltonian
generating the time evolution.
"""

from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.linalg import expm


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def nested_commutator(operators: List[np.ndarray]) -> np.ndarray:
    """
    Compute nested commutator [A, [B, [C, ...]]] from left to right.

    For [A, B, C]: returns [A, [B, C]]
    """
    if len(operators) < 2:
        raise ValueError("Need at least 2 operators for commutator")

    result = commutator(operators[-2], operators[-1])
    for op in reversed(operators[:-2]):
        result = commutator(op, result)
    return result


def first_order_term(H_func: Callable[[float], np.ndarray],
                     t_final: float,
                     n_points: int = 100) -> np.ndarray:
    """
    Compute the first-order Magnus term Omega_1(t).

    Omega_1 = -i * integral_0^t H(t') dt'

    Parameters
    ----------
    H_func : callable
        Function H(t) returning Hamiltonian at time t
    t_final : float
        Upper integration limit
    n_points : int
        Number of quadrature points

    Returns
    -------
    np.ndarray
        First-order Magnus term Omega_1
    """
    # Get dimension from H at t=0
    H0 = H_func(0.0)
    d = H0.shape[0]

    # Simple trapezoidal integration
    times = np.linspace(0, t_final, n_points)
    dt = times[1] - times[0]

    integral = np.zeros((d, d), dtype=complex)
    for i, t in enumerate(times):
        H_t = H_func(t)
        weight = 0.5 if (i == 0 or i == n_points - 1) else 1.0
        integral += weight * H_t * dt

    return -1j * integral


def second_order_term(H_func: Callable[[float], np.ndarray],
                      t_final: float,
                      n_points: int = 50) -> np.ndarray:
    """
    Compute the second-order Magnus term Omega_2(t).

    Omega_2 = -1/2 * integral_0^t dt1 integral_0^t1 dt2 [H(t1), H(t2)]

    Parameters
    ----------
    H_func : callable
        Function H(t) returning Hamiltonian at time t
    t_final : float
        Upper integration limit
    n_points : int
        Number of quadrature points per dimension

    Returns
    -------
    np.ndarray
        Second-order Magnus term Omega_2
    """
    H0 = H_func(0.0)
    d = H0.shape[0]

    # Double integral using nested trapezoidal rule
    # integral_0^t dt1 integral_0^t1 dt2 [H(t1), H(t2)]
    times = np.linspace(0, t_final, n_points)
    dt = times[1] - times[0]

    integral = np.zeros((d, d), dtype=complex)

    for i, t1 in enumerate(times):
        H_t1 = H_func(t1)
        weight1 = 0.5 if (i == 0 or i == n_points - 1) else 1.0

        # Inner integral from 0 to t1
        inner = np.zeros((d, d), dtype=complex)
        for j in range(i + 1):
            t2 = times[j]
            H_t2 = H_func(t2)
            weight2 = 0.5 if (j == 0 or j == i) else 1.0
            inner += weight2 * commutator(H_t1, H_t2) * dt

        integral += weight1 * inner * dt

    return -0.5 * integral


def third_order_term_placeholder(H_func: Callable[[float], np.ndarray],
                                  t_final: float,
                                  n_points: int = 30) -> np.ndarray:
    """
    Placeholder for third-order Magnus term Omega_3(t).

    Omega_3 = (1/6) * integral terms involving [H, [H, H]]

    The full formula involves:
    integral_0^t dt1 integral_0^t1 dt2 integral_0^t2 dt3
        ([H(t1), [H(t2), H(t3)]] + [H(t3), [H(t2), H(t1)]])

    TODO: Implement full third-order term

    Parameters
    ----------
    H_func : callable
        Hamiltonian function
    t_final : float
        Integration time
    n_points : int
        Quadrature points

    Returns
    -------
    np.ndarray
        Placeholder (zeros for now)
    """
    H0 = H_func(0.0)
    d = H0.shape[0]

    # TODO: Implement third-order term
    # For now, return zeros as placeholder
    return np.zeros((d, d), dtype=complex)


class MagnusExpansion:
    """
    Compute Magnus expansion to specified order.

    The Magnus expansion expresses the time evolution operator as:

        U(t) = exp(Omega(t))

    where Omega(t) = sum_{n=1}^{max_order} Omega_n(t)

    Parameters
    ----------
    max_order : int
        Maximum order of expansion (1, 2, or 3)
    n_points : int
        Number of quadrature points for integration

    Examples
    --------
    >>> def H(t):
    ...     return omega * sigma_x + delta(t) * sigma_z
    >>> magnus = MagnusExpansion(max_order=2)
    >>> Omega = magnus.compute(H, t_final=1.0)
    >>> U = magnus.unitary(H, t_final=1.0)
    """

    def __init__(self, max_order: int = 1, n_points: int = 100):
        if max_order < 1 or max_order > 3:
            raise ValueError("max_order must be 1, 2, or 3")

        self.max_order = max_order
        self.n_points = n_points

    def compute_term(self, H_func: Callable[[float], np.ndarray],
                     t_final: float, order: int) -> np.ndarray:
        """
        Compute a specific order term in the Magnus expansion.

        Parameters
        ----------
        H_func : callable
            Hamiltonian H(t)
        t_final : float
            Final time
        order : int
            Which order term (1, 2, or 3)

        Returns
        -------
        np.ndarray
            Omega_n term
        """
        if order == 1:
            return first_order_term(H_func, t_final, self.n_points)
        elif order == 2:
            return second_order_term(H_func, t_final, max(self.n_points // 2, 20))
        elif order == 3:
            return third_order_term_placeholder(H_func, t_final,
                                                 max(self.n_points // 3, 10))
        else:
            raise ValueError(f"Order {order} not implemented")

    def compute(self, H_func: Callable[[float], np.ndarray],
                t_final: float) -> np.ndarray:
        """
        Compute the total Magnus generator Omega up to max_order.

        Parameters
        ----------
        H_func : callable
            Hamiltonian H(t)
        t_final : float
            Final time

        Returns
        -------
        np.ndarray
            Omega = Omega_1 + Omega_2 + ...
        """
        Omega = self.compute_term(H_func, t_final, 1)

        for n in range(2, self.max_order + 1):
            Omega += self.compute_term(H_func, t_final, n)

        return Omega

    def unitary(self, H_func: Callable[[float], np.ndarray],
                t_final: float) -> np.ndarray:
        """
        Compute the unitary evolution operator via Magnus expansion.

        U(t) = exp(Omega(t))

        Parameters
        ----------
        H_func : callable
            Hamiltonian H(t)
        t_final : float
            Final time

        Returns
        -------
        np.ndarray
            Unitary matrix U(t)
        """
        Omega = self.compute(H_func, t_final)
        return expm(Omega)

    def compute_trajectory(self, H_func: Callable[[float], np.ndarray],
                           times: np.ndarray) -> List[np.ndarray]:
        """
        Compute Omega(t) at multiple time points.

        Parameters
        ----------
        H_func : callable
            Hamiltonian H(t)
        times : np.ndarray
            Array of time points

        Returns
        -------
        list of np.ndarray
            Omega(t) at each time
        """
        return [self.compute(H_func, t) for t in times]


# =============================================================================
# Noise perturbation expansion
# =============================================================================

def perturbed_magnus_first_order(H0_func: Callable[[float], np.ndarray],
                                  V_func: Callable[[float], np.ndarray],
                                  t_final: float,
                                  n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Magnus expansion for H = H_0 + epsilon * V to first order in epsilon.

    This separates the unperturbed evolution from the noise perturbation.

    Parameters
    ----------
    H0_func : callable
        Unperturbed Hamiltonian H_0(t)
    V_func : callable
        Noise operator V(t)
    t_final : float
        Final time
    n_points : int
        Quadrature points

    Returns
    -------
    tuple
        (Omega_0, Omega_V) where:
        - Omega_0 = -i * integral H_0(t) dt (unperturbed)
        - Omega_V = -i * integral V(t) dt (noise contribution)
    """
    Omega_0 = first_order_term(H0_func, t_final, n_points)
    Omega_V = first_order_term(V_func, t_final, n_points)
    return Omega_0, Omega_V


def perturbed_magnus_second_order(H0_func: Callable[[float], np.ndarray],
                                   V_func: Callable[[float], np.ndarray],
                                   t_final: float,
                                   n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Magnus expansion for H = H_0 + epsilon * V to second order.

    Returns terms organized by power of epsilon:
    - O(1): Omega_0^(2) from [H_0, H_0] (zero for time-independent H_0)
    - O(epsilon): Cross terms [H_0, V] and [V, H_0]
    - O(epsilon^2): Omega_V^(2) from [V, V]

    Parameters
    ----------
    H0_func : callable
        Unperturbed Hamiltonian
    V_func : callable
        Noise operator
    t_final : float
        Final time
    n_points : int
        Quadrature points

    Returns
    -------
    tuple
        (Omega_00, Omega_0V, Omega_VV) second-order contributions

    Notes
    -----
    The cross term Omega_0V contributes to the second-order filter function
    for detuning noise (when V ~ sigma_z) and amplitude noise (when V ~ sigma_x).

    TODO: Implement the full second-order cross terms. Currently placeholders.
    """
    H0 = H0_func(0.0)
    d = H0.shape[0]

    # Placeholder implementations
    # TODO: Implement proper second-order cross terms

    # Omega_00: [H_0, H_0] terms (usually zero or small)
    Omega_00 = second_order_term(H0_func, t_final, n_points)

    # Omega_0V: Cross terms [H_0, V] - this gives second-order filter function
    # TODO: Implement proper cross-term integration
    Omega_0V = np.zeros((d, d), dtype=complex)

    # Omega_VV: [V, V] terms (usually zero for single noise channel)
    Omega_VV = np.zeros((d, d), dtype=complex)

    return Omega_00, Omega_0V, Omega_VV
