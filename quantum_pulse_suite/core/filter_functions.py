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
        result[resonance_mask] = (1j * np.exp(1j * t * omega) *
                                   (1j - 1j * np.exp(2j * tau * omega) +
                                    2 * tau * omega) / (4 * omega))

    if np.any(off_resonance_mask):
        w_off = w[off_resonance_mask]
        num = (1j * np.exp(1j * w_off * t) *
               (w_off - 1j * np.exp(1j * w_off * tau) *
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

        for index, (f, g, theta, tau, t_end) in enumerate(self._poly_list):
            factor = self._compute_factor(frequencies, curr_time, tau)

            if index == 0:
                # First segment: only Fz contribution
                Fz_total += factor
            else:
                # Compute expressions for x, y, z components
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                # Expression: 2*cos(theta)*f*g + sin(theta)*(f^2 - conj(g)^2)
                expr = 2 * cos_theta * f * g + sin_theta * (f**2 - np.conj(g)**2)

                Fx_total += factor * np.real(1j * expr)
                Fy_total += factor * np.imag(1j * expr)

                # Fz: cos(theta)*(|f|^2 - |g|^2) - sin(theta)*(f*g + conj(f)*conj(g))
                fz_expr = (cos_theta * (f * np.conj(f) - g * np.conj(g)) -
                           sin_theta * (f * g + np.conj(f) * np.conj(g)))
                Fz_total += factor * fz_expr

            curr_time = t_end - tau + tau  # Update to end time of segment

        return np.real(Fx_total), np.real(Fy_total), np.real(Fz_total)


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
        Fx_total = np.zeros(len(frequencies), dtype=complex)
        Fy_total = np.zeros(len(frequencies), dtype=complex)
        Fz_total = np.zeros(len(frequencies), dtype=complex)

        curr_time = 0.0

        for f, g, omega, n_x, n_y, n_z, tau in self._poly_list:
            # Compute phi from axis components (for XY plane projection)
            phi = np.arctan2(-n_y, n_x) if (n_x != 0 or n_y != 0) else 0.0

            # Compute cj and sj for each frequency
            c_vals = np.array([cj(w, curr_time, omega, tau) for w in frequencies])
            s_vals = np.array([sj(w, curr_time, omega, tau) for w in frequencies])

            # Common expression for Fx, Fy
            expr = (c_vals * f * np.conj(g) +
                    s_vals * (np.exp(1j * phi) * f**2 +
                             np.exp(-1j * phi) * np.conj(g)**2))

            Fx_total += np.real(-1j * expr)
            Fy_total += np.imag(1j * expr)

            # Fz expression
            fz_expr = (c_vals * (f * np.conj(f) - g * np.conj(g)) -
                       s_vals * (np.exp(1j * phi) * f * g +
                                np.exp(-1j * phi) * np.conj(f) * np.conj(g)))
            Fz_total += fz_expr

            curr_time += tau

        return np.real(Fx_total), np.real(Fy_total), np.real(Fz_total)


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
