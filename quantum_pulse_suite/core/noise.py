"""
Noise generation utilities for quantum pulse sequence simulations.

This module provides functions and classes for generating time series
with specified spectral densities.
"""

from typing import Callable, Tuple, Optional
import numpy as np


def generate_time_series(
    psd: Callable[[np.ndarray], np.ndarray],
    n_points: int,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a time series with a given power spectral density.

    Uses the spectral method: generate white noise in the frequency domain,
    scale by sqrt(PSD), then transform back to the time domain.

    Parameters
    ----------
    psd : callable
        Power spectral density function S(w) that takes angular frequency
    n_points : int
        Number of time points to generate
    dt : float
        Time step between points

    Returns
    -------
    tuple
        (time_series, frequencies, generated_psd) where:
        - time_series: Real-valued noise time series
        - frequencies: Frequency array for PSD plot
        - generated_psd: Computed PSD of the generated series
    """
    # Generate white noise
    white_noise = np.random.normal(0, 1 / np.sqrt(dt), n_points)

    # Transform to frequency domain
    white_noise_fft = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(n_points, dt)

    # Convert to angular frequency and apply PSD scaling
    angular_freqs = 2 * np.pi * np.abs(freqs)
    scaling_factor = np.sqrt(psd(angular_freqs))
    scaling_factor[0] = 0  # Zero DC component to avoid divergence

    scaled_fft = white_noise_fft * scaling_factor

    # Transform back to time domain
    time_series = np.fft.ifft(scaled_fft)

    # Calculate the PSD of the generated time series (for verification)
    generated_fft = np.fft.fft(time_series)
    generated_psd = (np.abs(generated_fft)**2) / n_points

    return np.real(time_series), freqs, generated_psd


def noise_interpolation(
    values: np.ndarray,
    df: float,
    f: float
) -> float:
    """
    Linearly interpolate between values equally spaced in frequency.

    Wraps around at the end of the array for periodic interpolation.

    Parameters
    ----------
    values : np.ndarray
        Array of values equally spaced in frequency
    df : float
        Frequency spacing between values
    f : float
        Frequency at which to interpolate

    Returns
    -------
    float
        Interpolated value at frequency f
    """
    num_values = len(values)
    total_freq = num_values * df

    # Normalize frequency to be within [0, total_freq)
    f = f % total_freq

    # Find indices for interpolation
    index1 = int(f // df)
    index2 = (index1 + 1) % num_values

    # Compute interpolation fraction
    f1 = index1 * df
    fraction = (f - f1) / df

    # Linear interpolation
    return (1 - fraction) * values[index1] + fraction * values[index2]


class NoiseGenerator:
    """
    Class for generating noise time series with various spectral properties.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility
    """

    # Predefined noise types
    WHITE = 0
    PINK = 1  # 1/f
    BROWNIAN = 2  # 1/f^2
    VIOLET = -1  # f
    BLUE = -2  # f^2

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        n_points: int,
        dt: float,
        noise_type: int = WHITE,
        amplitude: float = 1.0,
        cutoff: float = 1e-14
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a noise time series with specified spectral properties.

        Parameters
        ----------
        n_points : int
            Number of time points
        dt : float
            Time step
        noise_type : int
            Power law exponent for S(w) ∝ 1/|w|^n:
            - 0: white noise
            - 1: 1/f (pink) noise
            - 2: 1/f^2 (Brownian) noise
            - Negative: blue noise (f^|n|)
        amplitude : float
            Noise amplitude scaling
        cutoff : float
            Low-frequency cutoff to prevent divergence

        Returns
        -------
        tuple
            (time_series, times) arrays
        """
        def psd(w):
            w = np.asarray(w)
            if noise_type >= 0:
                return amplitude / (np.abs(w)**noise_type + cutoff)
            else:
                # Blue noise: increases with frequency
                return amplitude * (np.abs(w)**(-noise_type) + cutoff)

        # Generate white noise with internal RNG
        white_noise = self._rng.normal(0, 1 / np.sqrt(dt), n_points)

        # Transform to frequency domain
        white_noise_fft = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(n_points, dt)

        # Apply PSD scaling
        angular_freqs = 2 * np.pi * np.abs(freqs)
        scaling_factor = np.sqrt(psd(angular_freqs))
        scaling_factor[0] = 0

        scaled_fft = white_noise_fft * scaling_factor
        time_series = np.real(np.fft.ifft(scaled_fft))

        times = np.arange(n_points) * dt
        return time_series, times

    def generate_correlated(
        self,
        n_points: int,
        dt: float,
        correlation_time: float,
        amplitude: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate noise with exponential correlation (Ornstein-Uhlenbeck process).

        The PSD is Lorentzian: S(w) = A * gamma / (w^2 + gamma^2)
        where gamma = 1/correlation_time.

        Parameters
        ----------
        n_points : int
            Number of time points
        dt : float
            Time step
        correlation_time : float
            Characteristic correlation time (1/gamma)
        amplitude : float
            Noise amplitude

        Returns
        -------
        tuple
            (time_series, times) arrays
        """
        gamma = 1.0 / correlation_time

        def psd(w):
            w = np.asarray(w)
            return amplitude * gamma / (w**2 + gamma**2)

        white_noise = self._rng.normal(0, 1 / np.sqrt(dt), n_points)
        white_noise_fft = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(n_points, dt)

        angular_freqs = 2 * np.pi * np.abs(freqs)
        scaling_factor = np.sqrt(psd(angular_freqs))
        scaling_factor[0] = 0

        scaled_fft = white_noise_fft * scaling_factor
        time_series = np.real(np.fft.ifft(scaled_fft))

        times = np.arange(n_points) * dt
        return time_series, times

    def generate_custom(
        self,
        n_points: int,
        dt: float,
        psd: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate noise with a custom power spectral density.

        Parameters
        ----------
        n_points : int
            Number of time points
        dt : float
            Time step
        psd : callable
            Power spectral density function S(w)

        Returns
        -------
        tuple
            (time_series, times) arrays
        """
        white_noise = self._rng.normal(0, 1 / np.sqrt(dt), n_points)
        white_noise_fft = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(n_points, dt)

        angular_freqs = 2 * np.pi * np.abs(freqs)
        scaling_factor = np.sqrt(psd(angular_freqs))
        scaling_factor[0] = 0

        scaled_fft = white_noise_fft * scaling_factor
        time_series = np.real(np.fft.ifft(scaled_fft))

        times = np.arange(n_points) * dt
        return time_series, times
