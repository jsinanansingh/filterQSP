"""
Allan variance computation for atomic clock experiments.

This module provides functions for computing the Allan variance of clock
measurements using the filter function formalism. The Dick effect, which
arises from dead time between interrogation cycles, is included.

Theory
------
For a clock with interrogation sequence characterized by filter function F(ω),
cycle time T_c = T_interrogation + T_dead, and local oscillator noise S_y(f),
the Allan variance is:

    σ²_y(τ) = Σ_m |g_m|² S_y(m/T_c)

where g_m are the Fourier coefficients of the sensitivity function, or
equivalently:

    σ²_y(τ) = (2/τ) ∫₀^∞ S_y(f) |H(f,τ)|² |G(f)|² df

where G(f) is the transfer function including the dead time effect.

References
----------
- Dick, G. J. (1987). "Local oscillator induced instabilities in trapped ion
  frequency standards." Proc. 19th PTTI Meeting.
- Santarelli, G. et al. (1998). "Frequency stability degradation of an
  oscillator slaved to a periodically interrogated atomic resonator."
  IEEE Trans. UFFC 45(4), 887-894.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy.integrate import simpson

from ..core.filter_functions import FilterFunction


def compute_sensitivity_function(
    ff: FilterFunction,
    t_interrogation: float,
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the time-domain sensitivity function g(t) from the filter function.

    The sensitivity function describes how phase noise at time t affects
    the final measurement. It is related to the filter function by:
    F(ω) = ∫ g(t) exp(iωt) dt

    Parameters
    ----------
    ff : FilterFunction
        Filter function calculator for the interrogation sequence
    t_interrogation : float
        Total interrogation time
    n_points : int
        Number of time points for discretization

    Returns
    -------
    tuple
        (times, g_t) where times is the time array and g_t is the
        sensitivity function values
    """
    times = np.linspace(0, t_interrogation, n_points)
    dt = times[1] - times[0]

    # Use inverse Fourier transform of F(ω) to get g(t)
    # Sample frequencies for inverse FFT
    freq_max = n_points / (2 * t_interrogation)
    frequencies = np.linspace(0.01, freq_max * 2 * np.pi, n_points)

    # Get filter function (use Fz component for dephasing noise)
    Fx, Fy, Fz = ff.filter_function(frequencies)

    # For dephasing, the sensitivity is primarily in z
    F_total = Fz

    # Inverse FFT to get time-domain sensitivity
    # This is an approximation; exact inversion would require careful treatment
    g_t = np.fft.ifft(np.fft.ifftshift(F_total)).real
    g_t = g_t[:n_points] * len(frequencies) * (frequencies[1] - frequencies[0]) / (2 * np.pi)

    return times, g_t


def dick_effect_coefficients(
    ff: FilterFunction,
    t_interrogation: float,
    t_dead: float,
    n_harmonics: int = 50,
    freq_points: int = 500
) -> np.ndarray:
    """
    Compute the Dick effect coefficients g_m.

    The Dick effect coefficients determine how the local oscillator noise
    at harmonics of the cycle frequency aliases into the clock stability.

    g_m = (1/T_c) ∫₀^T_c g(t) exp(-2πi m t / T_c) dt

    where g(t) is the sensitivity function (zero during dead time).

    Parameters
    ----------
    ff : FilterFunction
        Filter function calculator for the interrogation sequence
    t_interrogation : float
        Interrogation time (Ramsey/Rabi duration)
    t_dead : float
        Dead time between interrogation cycles
    n_harmonics : int
        Number of harmonics to compute (m = 1, 2, ..., n_harmonics)
    freq_points : int
        Number of frequency points for filter function evaluation

    Returns
    -------
    np.ndarray
        Array of |g_m|² values for m = 1 to n_harmonics
    """
    t_cycle = t_interrogation + t_dead

    # Compute g_m from the filter function
    # g_m = F(2πm/T_c) / T_c for the interrogation window
    g_m_squared = np.zeros(n_harmonics)

    for m in range(1, n_harmonics + 1):
        omega_m = 2 * np.pi * m / t_cycle

        # Get filter function at this harmonic
        Fx, Fy, Fz = ff.filter_function(np.array([omega_m]))

        # Total sensitivity (squared)
        F_squared = np.abs(Fx[0])**2 + np.abs(Fy[0])**2 + np.abs(Fz[0])**2

        # Dick coefficient includes the duty cycle factor
        duty_cycle = t_interrogation / t_cycle
        g_m_squared[m - 1] = F_squared / t_cycle**2

    return g_m_squared


def allan_variance_dick(
    ff: FilterFunction,
    psd_func: Callable[[np.ndarray], np.ndarray],
    t_interrogation: float,
    t_dead: float,
    tau: float,
    n_harmonics: int = 100
) -> float:
    """
    Compute Allan variance including the Dick effect.

    For a clock with periodic interrogation, the Allan variance is:

    σ²_y(τ) = (1/τ) Σ_{m=1}^∞ |g_m|² S_y(m f_c)

    where f_c = 1/T_c is the cycle frequency and g_m are the Dick coefficients.

    Parameters
    ----------
    ff : FilterFunction
        Filter function calculator for the interrogation sequence
    psd_func : callable
        Fractional frequency noise PSD function S_y(f) that takes frequency
        in Hz and returns the one-sided PSD
    t_interrogation : float
        Interrogation time
    t_dead : float
        Dead time between cycles
    tau : float
        Averaging time for Allan variance
    n_harmonics : int
        Number of harmonics to include in sum

    Returns
    -------
    float
        Allan variance σ²_y(τ)
    """
    t_cycle = t_interrogation + t_dead
    f_cycle = 1.0 / t_cycle

    # Get Dick coefficients
    g_m_sq = dick_effect_coefficients(ff, t_interrogation, t_dead, n_harmonics)

    # Sum over harmonics
    variance = 0.0
    for m in range(1, n_harmonics + 1):
        f_m = m * f_cycle
        # Convert to angular frequency for PSD if needed
        S_y = psd_func(np.array([2 * np.pi * f_m]))[0]
        variance += g_m_sq[m - 1] * S_y

    # Scale by averaging time
    # For τ >> T_c, the Allan variance scales as 1/τ
    n_cycles = int(tau / t_cycle)
    if n_cycles < 1:
        n_cycles = 1

    return variance / n_cycles


def allan_variance_continuous(
    ff: FilterFunction,
    psd_func: Callable[[np.ndarray], np.ndarray],
    t_interrogation: float,
    t_dead: float,
    tau: float,
    freq_range: Tuple[float, float] = (1e-3, 1e6),
    n_points: int = 1000
) -> float:
    """
    Compute Allan variance using continuous integration.

    This method integrates the product of the filter function and noise PSD
    over all frequencies, accounting for the averaging time τ:

    σ²_y(τ) = (2/τ) ∫₀^∞ S_y(f) |sin(πfτ)/(πfτ)|² |F(2πf)|² / T_c² df

    Parameters
    ----------
    ff : FilterFunction
        Filter function calculator
    psd_func : callable
        Fractional frequency noise PSD S_y(f) in Hz
    t_interrogation : float
        Interrogation time
    t_dead : float
        Dead time
    tau : float
        Averaging time
    freq_range : tuple
        (f_min, f_max) frequency range for integration in Hz
    n_points : int
        Number of points for integration

    Returns
    -------
    float
        Allan variance σ²_y(τ)
    """
    t_cycle = t_interrogation + t_dead

    # Frequency array (Hz)
    f = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
    omega = 2 * np.pi * f

    # Get filter function
    susceptibility = ff.noise_susceptibility(omega)

    # Noise PSD (expects angular frequency)
    S_y = psd_func(omega)

    # Allan variance transfer function: |sin(πfτ)/(πfτ)|²
    x = np.pi * f * tau
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        H_allan = np.where(np.abs(x) < 1e-10, 1.0, (np.sin(x) / x)**2)

    # Duty cycle correction
    duty_cycle = t_interrogation / t_cycle

    # Integrand
    integrand = S_y * H_allan * susceptibility / t_cycle**2

    # Use log-spaced integration
    # Convert to linear spacing for simpson
    df = np.diff(f)
    integral = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * df)

    return 2 * integral / tau


def allan_deviation(
    ff: FilterFunction,
    psd_func: Callable[[np.ndarray], np.ndarray],
    t_interrogation: float,
    t_dead: float,
    tau: float,
    method: str = 'dick',
    **kwargs
) -> float:
    """
    Compute Allan deviation (square root of Allan variance).

    Parameters
    ----------
    ff : FilterFunction
        Filter function calculator
    psd_func : callable
        Fractional frequency noise PSD
    t_interrogation : float
        Interrogation time
    t_dead : float
        Dead time
    tau : float
        Averaging time
    method : str
        'dick' for harmonic sum method, 'continuous' for integral method
    **kwargs
        Additional arguments passed to the variance function

    Returns
    -------
    float
        Allan deviation σ_y(τ)
    """
    if method == 'dick':
        var = allan_variance_dick(ff, psd_func, t_interrogation, t_dead, tau, **kwargs)
    elif method == 'continuous':
        var = allan_variance_continuous(ff, psd_func, t_interrogation, t_dead, tau, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dick' or 'continuous'.")

    return np.sqrt(var)


def allan_variance_vs_tau(
    ff: FilterFunction,
    psd_func: Callable[[np.ndarray], np.ndarray],
    t_interrogation: float,
    t_dead: float,
    tau_array: np.ndarray,
    method: str = 'dick',
    **kwargs
) -> np.ndarray:
    """
    Compute Allan variance as a function of averaging time.

    Parameters
    ----------
    ff : FilterFunction
        Filter function calculator
    psd_func : callable
        Fractional frequency noise PSD
    t_interrogation : float
        Interrogation time
    t_dead : float
        Dead time
    tau_array : np.ndarray
        Array of averaging times
    method : str
        'dick' or 'continuous'
    **kwargs
        Additional arguments

    Returns
    -------
    np.ndarray
        Allan variance at each tau value
    """
    variances = np.zeros_like(tau_array)

    for i, tau in enumerate(tau_array):
        if method == 'dick':
            variances[i] = allan_variance_dick(
                ff, psd_func, t_interrogation, t_dead, tau, **kwargs
            )
        else:
            variances[i] = allan_variance_continuous(
                ff, psd_func, t_interrogation, t_dead, tau, **kwargs
            )

    return variances


def quantum_projection_noise_limit(
    t_interrogation: float,
    t_dead: float,
    n_atoms: int,
    transition_frequency: float
) -> Callable[[float], float]:
    """
    Compute the quantum projection noise (QPN) limited Allan deviation.

    The QPN limit for an atomic clock is:

    σ_y(τ) = 1 / (ω₀ √(N T_R τ))

    where ω₀ is the transition frequency, N is the atom number,
    T_R is the Ramsey time, and τ is the averaging time.

    Parameters
    ----------
    t_interrogation : float
        Interrogation (Ramsey) time T_R
    t_dead : float
        Dead time (for duty cycle correction)
    n_atoms : int
        Number of atoms
    transition_frequency : float
        Clock transition frequency in Hz

    Returns
    -------
    callable
        Function that returns σ_y(τ) for given τ
    """
    t_cycle = t_interrogation + t_dead
    duty_cycle = t_interrogation / t_cycle
    omega_0 = 2 * np.pi * transition_frequency

    def qpn_limit(tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """QPN-limited Allan deviation."""
        return 1.0 / (omega_0 * np.sqrt(n_atoms * t_interrogation * tau * duty_cycle))

    return qpn_limit
