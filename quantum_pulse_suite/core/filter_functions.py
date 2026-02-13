"""
Filter function classes for quantum signal processing.

This module provides filter function calculators for both instantaneous
and continuous pulse sequences, used for computing noise susceptibility.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Union
import numpy as np


def cj(w: Union[float, np.ndarray], t: float, omega: float, tau: float,
       eps: float = 1e-8) -> Union[complex, np.ndarray]:
    """
    Compute the cosine Fourier integral for continuous pulse filter functions.

    This function computes the integral involving cos(omega*t') weighted by
    exp(i*w*t') over the pulse duration, with special handling for resonance.

    Parameters
    ----------
    w : float or np.ndarray
        Angular frequency at which to evaluate
    t : float
        Start time of the pulse segment
    omega : float
        Rabi frequency of the pulse
    tau : float
        Duration of the pulse
    eps : float
        Tolerance for resonance condition |w - omega| < eps

    Returns
    -------
    complex or np.ndarray
        Value of the cj integral
    """
    w = np.asarray(w)
    scalar_input = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=complex)

    # Handle resonance case
    resonance_mask = np.abs(w - omega) < eps
    off_resonance_mask = ~resonance_mask

    if np.any(resonance_mask):
        # Use the limiting form at resonance
        result[resonance_mask] = (np.exp(1j * t * omega) *
                                   (1j - 1j * np.exp(2j * tau * omega) +
                                    2 * tau * omega) / (4 * omega))

    if np.any(off_resonance_mask):
        w_off = w[off_resonance_mask]
        num = (1j * np.exp(1j * w_off * t) *
               (w_off - np.exp(1j * w_off * tau) *
                (w_off * np.cos(omega * tau) - 1j * omega * np.sin(omega * tau))))
        den = w_off**2 - omega**2
        result[off_resonance_mask] = num / den

    return result.item() if scalar_input else result


def sj(w: Union[float, np.ndarray], t: float, omega: float, tau: float,
       eps: float = 1e-8) -> Union[complex, np.ndarray]:
    """
    Compute the sine Fourier integral for continuous pulse filter functions.

    This function computes the integral involving sin(omega*t') weighted by
    exp(i*w*t') over the pulse duration, with special handling for resonance.

    Parameters
    ----------
    w : float or np.ndarray
        Angular frequency at which to evaluate
    t : float
        Start time of the pulse segment
    omega : float
        Rabi frequency of the pulse
    tau : float
        Duration of the pulse
    eps : float
        Tolerance for resonance condition |w - omega| < eps

    Returns
    -------
    complex or np.ndarray
        Value of the sj integral
    """
    w = np.asarray(w)
    scalar_input = w.ndim == 0
    w = np.atleast_1d(w)

    result = np.zeros_like(w, dtype=complex)

    # Handle resonance case
    resonance_mask = np.abs(w - omega) < eps
    off_resonance_mask = ~resonance_mask

    if np.any(resonance_mask):
        # Use the limiting form at resonance
        result[resonance_mask] = (np.exp(1j * t * omega) *
                                   (1 - np.exp(2j * tau * omega) +
                                    2j * tau * omega) / (4 * omega))

    if np.any(off_resonance_mask):
        w_off = w[off_resonance_mask]
        num = (np.exp(1j * w_off * t) *
               (-omega + np.exp(1j * w_off * tau) *
                (omega * np.cos(omega * tau) - 1j * w_off * np.sin(omega * tau))))
        den = w_off**2 - omega**2
        result[off_resonance_mask] = num / den

    return result.item() if scalar_input else result


class FilterFunction(ABC):
    """Abstract base class for filter function calculators."""

    @abstractmethod
    def filter_function(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the filter function at given frequencies.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies at which to evaluate

        Returns
        -------
        tuple
            (Fx, Fy, Fz) filter function components
        """
        pass

    def noise_susceptibility(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute total noise susceptibility |F(w)|^2.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies at which to evaluate

        Returns
        -------
        np.ndarray
            Total filter function magnitude squared
        """
        Fx, Fy, Fz = self.filter_function(frequencies)
        return np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2


class InstantaneousFilterFunction(FilterFunction):
    """
    Filter function calculator for instantaneous pulse sequences.

    Uses the exponential factor formula:
    factor = 1j * exp(iw*t0) * (exp(-iw*τ) - 1) / w

    Parameters
    ----------
    poly_list : list
        List of polynomial tuples (f, g, theta, tau, t_end) from
        InstantaneousPulseSequence.compute_polynomials()
    """

    def __init__(self, poly_list: List[Tuple]):
        self._poly_list = poly_list

    def _compute_factor(self, w: np.ndarray, t0: float, tau: float) -> np.ndarray:
        """Compute the exponential factor for instantaneous filter functions."""
        # Handle w=0 case to avoid division by zero
        w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return 1j * np.exp(1j * w * t0) * (np.exp(-1j * w_safe * tau) - 1) / w_safe

    def filter_function(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute filter function components at given frequencies.

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies

        Returns
        -------
        tuple
            (Fx, Fy, Fz) arrays
        """
        frequencies = np.asarray(frequencies)
        Fx_total = np.zeros_like(frequencies, dtype=complex)
        Fy_total = np.zeros_like(frequencies, dtype=complex)
        Fz_total = np.zeros_like(frequencies, dtype=complex)

        curr_time = 0.0

        for f, g, theta, tau, t_end in self._poly_list:
            factor = self._compute_factor(frequencies, curr_time, tau)

            # Compute Bloch component expressions from Cayley-Klein params
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Rx - iRy = -i(2cos(θ)fg* + sin(θ)(f² - g*²))  [paper eq (41)]
            expr = 2 * cos_theta * f * np.conj(g) + sin_theta * (f**2 - np.conj(g)**2)

            Fx_total += factor * np.real(-1j * expr)
            Fy_total += factor * np.imag(1j * expr)

            # Rz = cos(θ)(|f|² - |g|²) - sin(θ)(fg + f*g*)
            fz_expr = (cos_theta * (f * np.conj(f) - g * np.conj(g)) -
                       sin_theta * (f * g + np.conj(f) * np.conj(g)))
            Fz_total += factor * fz_expr

            curr_time = t_end

        return Fx_total, Fy_total, Fz_total


class ContinuousFilterFunction(FilterFunction):
    """
    Filter function calculator for continuous pulse sequences.

    Uses cj() and sj() Fourier integrals with the rotation axis
    to compute filter function components.

    Parameters
    ----------
    poly_list : list
        List of polynomial tuples (f, g, omega, n_x, n_y, n_z, tau) from
        ContinuousPulseSequence.compute_polynomials()
    """

    def __init__(self, poly_list: List[Tuple]):
        self._poly_list = poly_list

    def _compute_z_and_fz(self, frequencies: np.ndarray):
        """Compute Z(+w), Z(-w), and Fz(w) for the filter function.

        Z(w) = R_x(w) - i*R_y(w) is the combined xy Fourier component.
        Uses c_j(-w) = conj(c_j(w)) to evaluate at negative frequencies
        without extra integration.

        Returns (Z_plus, Z_minus, Fz) complex arrays.
        """
        n = len(frequencies)
        Z_plus = np.zeros(n, dtype=complex)
        Z_minus = np.zeros(n, dtype=complex)
        Fz_total = np.zeros(n, dtype=complex)

        curr_time = 0.0

        for f, g, omega, n_x, n_y, n_z, tau in self._poly_list:
            phi = np.arctan2(-n_y, n_x) if (n_x != 0 or n_y != 0) else 0.0
            phase_p = np.exp(1j * phi)
            phase_m = np.exp(-1j * phi)

            # Fourier integrals at +w
            c_plus = np.array([cj(w, curr_time, omega, tau) for w in frequencies])
            s_plus = np.array([sj(w, curr_time, omega, tau) for w in frequencies])

            # At -w: c_j(-w) = conj(c_j(w)) for real cos/sin kernels
            c_minus = np.conj(c_plus)
            s_minus = np.conj(s_plus)

            # Rx-iRy expression: -i * expr
            # Sign on g*² term matches instantaneous convention (f²-g*²)
            # because code tracks U_code = U_phys†, not U_phys
            xy_common = phase_p * f**2 - phase_m * np.conj(g)**2
            expr_plus = 2 * c_plus * f * np.conj(g) + s_plus * xy_common
            expr_minus = 2 * c_minus * f * np.conj(g) + s_minus * xy_common

            Z_plus += -1j * expr_plus
            Z_minus += -1j * expr_minus

            # Fz: direct Fourier transform of Rz(t) [paper eq (44)]
            fz_common = phase_p * f * g + phase_m * np.conj(f) * np.conj(g)
            Fz_total += (c_plus * (f * np.conj(f) - g * np.conj(g)) -
                         s_plus * fz_common)

            curr_time += tau

        return Z_plus, Z_minus, Fz_total

    def filter_function(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute filter function Bloch components at given frequencies.

        For continuous pulses, R_k(t) oscillates within segments, so
        recovering individual R_x(w) and R_y(w) requires evaluating the
        combined Z(w) = R_x(w) - i*R_y(w) at both +w and -w:
            R_x(w) = (Z(w) + Z(-w)*) / 2
            R_y(w) = i*(Z(w) - Z(-w)*) / 2

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies

        Returns
        -------
        tuple
            (Fx, Fy, Fz) complex arrays — the Bloch-component Fourier
            transforms R_x(w), R_y(w), R_z(w)
        """
        frequencies = np.asarray(frequencies)
        Z_plus, Z_minus, Fz = self._compute_z_and_fz(frequencies)

        # Recover individual components from Z(+w) and Z(-w)
        # Using: R_k(-w) = R_k(w)* for real R_k(t)
        Z_neg_star = np.conj(Z_minus)  # Z(-w)* = R_x(w) + i*R_y(w)
        Fx = (Z_plus + Z_neg_star) / 2
        Fy = 1j * (Z_plus - Z_neg_star) / 2

        return Fx, Fy, Fz

    def noise_susceptibility(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute total noise susceptibility |F(w)|^2.

        Uses the identity:
            |R_x(w)|^2 + |R_y(w)|^2 = (|Z(w)|^2 + |Z(-w)|^2) / 2

        Parameters
        ----------
        frequencies : np.ndarray
            Angular frequencies

        Returns
        -------
        np.ndarray
            Total filter function magnitude squared
        """
        frequencies = np.asarray(frequencies)
        Z_plus, Z_minus, Fz = self._compute_z_and_fz(frequencies)
        return (np.abs(Z_plus)**2 + np.abs(Z_minus)**2) / 2 + np.abs(Fz)**2


class ColoredNoisePSD:
    """
    Factory class for common colored noise power spectral densities.

    Provides static methods to create PSD functions for various noise types.
    """

    @staticmethod
    def white_noise(amplitude: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a white noise PSD (flat spectrum).

        Parameters
        ----------
        amplitude : float
            Noise amplitude

        Returns
        -------
        callable
            PSD function S(w)
        """
        def psd(w):
            return amplitude * np.ones_like(w)
        return psd

    @staticmethod
    def one_over_f(amplitude: float = 1.0, cutoff: float = 1e-14) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a 1/f (pink) noise PSD.

        Parameters
        ----------
        amplitude : float
            Noise amplitude
        cutoff : float
            Low-frequency cutoff to avoid divergence

        Returns
        -------
        callable
            PSD function S(w)
        """
        def psd(w):
            w = np.asarray(w)
            return amplitude / (np.abs(w) + cutoff)
        return psd

    @staticmethod
    def one_over_f2(amplitude: float = 1.0, cutoff: float = 1e-14) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a 1/f^2 (Brownian) noise PSD.

        Parameters
        ----------
        amplitude : float
            Noise amplitude
        cutoff : float
            Low-frequency cutoff to avoid divergence

        Returns
        -------
        callable
            PSD function S(w)
        """
        def psd(w):
            w = np.asarray(w)
            return amplitude / (np.abs(w)**2 + cutoff)
        return psd

    @staticmethod
    def lorentzian(amplitude: float = 1.0, gamma: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a Lorentzian noise PSD.

        S(w) = A * gamma / (w^2 + gamma^2)

        Parameters
        ----------
        amplitude : float
            Noise amplitude
        gamma : float
            Width parameter (correlation rate)

        Returns
        -------
        callable
            PSD function S(w)
        """
        def psd(w):
            w = np.asarray(w)
            return amplitude * gamma / (w**2 + gamma**2)
        return psd

    @staticmethod
    def generic_power_law(amplitude: float = 1.0, exponent: float = 1.0,
                          cutoff: float = 1e-14) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a generic power-law noise PSD: S(w) = A / |w|^n

        Parameters
        ----------
        amplitude : float
            Noise amplitude
        exponent : float
            Power law exponent
        cutoff : float
            Low-frequency cutoff

        Returns
        -------
        callable
            PSD function S(w)
        """
        def psd(w):
            w = np.asarray(w)
            return amplitude / (np.abs(w)**exponent + cutoff)
        return psd
