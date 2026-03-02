"""
Pulse sequence optimizer for equiangular continuous-drive sequences.

Searches for the N-segment equiangular pulse sequence that maximises

    FOM = |partial_delta <M>|^2 / Kubo_variance(S)

i.e. the clock's Fisher information for frequency estimation, for a
given noise power spectral density S(omega).

Parameterisation
----------------
* N continuous-drive segments of equal duration tau = T / N  (equiangular)
* Each segment is driven at a shared Rabi frequency Omega with drive axis
  (cos(phi_k), sin(phi_k), 0) in the xy-plane.
* Free parameters: Omega > 0 and phases phi_2, ..., phi_N  (phi_1 = 0 by
  gauge choice, reducing the redundancy from global phase freedom).

The objective minimised is Kubo_variance / sensitivity_sq, which equals
the Cramer-Rao lower bound on the single-shot frequency estimation variance.

Noise PSDs
----------
* white   : S(omega) = S0  (constant)
* 1/f     : S(omega) = S0 / |omega|  (1/f noise; integrated from the lowest
            FFT frequency, well above any 1/f corner expected in practice)

Both PSDs lie in the regime where the Kubo linear-response approximation
is valid (small noise perturbation).

Algorithm
---------
1. Global search with scipy differential_evolution over (Omega, phi_2, ..., phi_N).
2. Local polishing with L-BFGS-B from n_restarts random starts.
3. Best result across all runs is returned.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Union
from scipy.integrate import simpson
from scipy.optimize import differential_evolution, minimize

from ..core.multilevel import MultiLevelPulseSequence
from ..core.three_level_filter import (
    fft_three_level_filter,
    detuning_sensitivity,
)


# =============================================================================
# Noise PSD presets
# =============================================================================

def white_noise_psd(amplitude: float = 1.0) -> Callable:
    """Flat (white) noise PSD: S(omega) = amplitude."""
    def S(w):
        return amplitude * np.ones_like(np.asarray(w, dtype=float))
    return S


def one_over_f_psd(amplitude: float = 1.0, cutoff: float = 1e-4) -> Callable:
    """
    1/f noise PSD: S(omega) = amplitude / max(|omega|, cutoff).

    The cutoff prevents a divergence at omega = 0; it should be well below
    the lowest filter function frequency (2*pi / (pad_factor * T)).
    """
    def S(w):
        w = np.asarray(w, dtype=float)
        return amplitude / np.maximum(np.abs(w), cutoff)
    return S


_PSD_PRESETS = {
    'white': white_noise_psd,
    '1/f':   one_over_f_psd,
}


# =============================================================================
# Sequence builder
# =============================================================================

def build_equiangular_3level(
    system,
    T: float,
    N: int,
    omega: float,
    phases,
) -> MultiLevelPulseSequence:
    """
    Build an N-segment equiangular MultiLevelPulseSequence.

    Each segment has duration T/N and drives the probe transition at
    Rabi frequency omega along axis (cos(phi_k), sin(phi_k), 0).

    Parameters
    ----------
    system : ThreeLevelClock
    T : float
        Total interrogation time.
    N : int
        Number of segments.
    omega : float
        Rabi frequency (rad/s), shared across all segments.
    phases : array_like, shape (N,)
        Drive phase angles phi_k (rad).

    Returns
    -------
    MultiLevelPulseSequence
        Sequence with polynomials already computed.
    """
    tau = T / N
    seq = MultiLevelPulseSequence(system, system.probe)
    for phi in phases:
        phi = float(phi)
        seq.add_continuous_pulse(omega, [np.cos(phi), np.sin(phi), 0.0], 0.0, tau)
    seq.compute_polynomials()
    return seq


# =============================================================================
# Single-sequence evaluation
# =============================================================================

def evaluate_sequence(
    seq: MultiLevelPulseSequence,
    S_func: Callable,
    m_y: float = 1.0,
    M=None,
    psi0=None,
    n_fft: int = 2048,
    pad_factor: int = 4,
):
    """
    Evaluate a 3-level probe sequence for clock optimisation.

    Computes the detuning sensitivity and Kubo variance in a single pass,
    using the FFT-based filter function for speed.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
        Probe sequence with compute_polynomials() already called.
    S_func : callable
        Noise PSD S(omega).
    m_y : float
        Measurement weight (for Fe = m_y^2 |Chi|^2).
    M : np.ndarray, optional
        Full measurement observable.  Default sigma_y^{gm}.
    psi0 : np.ndarray, optional
        Initial state.  Default (|g> + |m>)/sqrt(2).
    n_fft : int
    pad_factor : int

    Returns
    -------
    sensitivity_sq : float
        |partial_delta <M>|^2 at delta=0.
    kubo_var : float
        integral Fe(omega) S(omega) domega / (2 pi).
    fom : float
        sensitivity_sq / kubo_var  (0 if either is non-positive).
    """
    _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)

    freqs, Fe, _, _ = fft_three_level_filter(
        seq, n_samples=n_fft, pad_factor=pad_factor, m_y=m_y)
    kubo_var = float(simpson(Fe * S_func(freqs), x=freqs) / (2.0 * np.pi))

    if sens_sq > 0.0 and kubo_var > 0.0:
        fom = sens_sq / kubo_var
    else:
        fom = 0.0

    return sens_sq, kubo_var, fom


# =============================================================================
# Result container
# =============================================================================

@dataclass
class PulseOptimizationResult:
    """
    Result of an equiangular pulse optimisation.

    Attributes
    ----------
    omega : float
        Optimal shared Rabi frequency (rad/s).
    phases : np.ndarray, shape (N,)
        Optimal drive phases phi_k (rad); phi[0] = 0 by convention.
    sensitivity_sq : float
        |partial_delta <M>|^2 at the optimum.
    kubo_var : float
        Kubo variance integral Fe(omega) S(omega) domega / (2 pi).
    fom : float
        Fisher information proxy: sensitivity_sq / kubo_var.
    noise_label : str
        Label of the noise PSD used.
    seq : MultiLevelPulseSequence
        The optimised sequence.
    """
    omega: float
    phases: np.ndarray
    sensitivity_sq: float
    kubo_var: float
    fom: float
    noise_label: str
    seq: object


# =============================================================================
# Optimiser
# =============================================================================

def optimize_equiangular_sequence(
    system,
    T: float,
    N: int,
    noise_psd: Union[str, Callable] = 'white',
    omega_max: Optional[float] = None,
    m_y: float = 1.0,
    M=None,
    psi0=None,
    n_restarts: int = 5,
    seed: int = 0,
    n_fft: int = 2048,
    pad_factor: int = 4,
    popsize: int = 15,
    tol: float = 1e-5,
    maxiter: int = 300,
) -> PulseOptimizationResult:
    """
    Optimise an equiangular continuous-drive 3-level clock sequence.

    Minimises the frequency estimation variance
        sigma^2_freq = Kubo_variance(S) / |partial_delta <M>|^2
    over N equal-duration continuous-drive segments, parametrised by a
    shared Rabi frequency Omega and N-1 free drive phases (phi_1 = 0).

    Parameters
    ----------
    system : ThreeLevelClock
    T : float
        Total interrogation time.
    N : int
        Number of equiangular segments (each of duration T/N).
    noise_psd : str or callable
        'white', '1/f', or a callable S(omega).
    omega_max : float, optional
        Upper bound on Omega.  Default: N * pi / T (one pi-pulse per segment).
    m_y : float
        Measurement weight m_y for Fe computation (default 1.0).
    M : np.ndarray, optional
        Full 3x3 measurement observable (default sigma_y^{gm}).
    psi0 : np.ndarray, optional
        Initial state for sensitivity (default (|g>+|m>)/sqrt(2)).
    n_restarts : int
        Extra random restarts for local polishing after global DE.
    seed : int
        Random seed.
    n_fft : int
        FFT sample count for filter function (default 2048).
    pad_factor : int
        FFT zero-padding factor (default 4).
    popsize : int
        DE population multiplier (default 15).
    tol : float
        DE convergence tolerance (default 1e-5).
    maxiter : int
        DE maximum generations (default 300).

    Returns
    -------
    PulseOptimizationResult
    """
    rng = np.random.default_rng(seed)

    if omega_max is None:
        omega_max = N * np.pi / T   # maximum: one pi-pulse per segment

    # Resolve noise PSD
    if isinstance(noise_psd, str):
        if noise_psd not in _PSD_PRESETS:
            raise ValueError(
                f"noise_psd must be 'white', '1/f', or a callable; got {noise_psd!r}")
        S_func = _PSD_PRESETS[noise_psd]()
        noise_label = noise_psd
    else:
        S_func = noise_psd
        noise_label = 'custom'

    # Parameter layout: x = [omega, phi_2, ..., phi_N]   (N free params total)
    # phi_1 = 0 is fixed to break the global-phase gauge redundancy.
    omega_lo = np.pi / (10.0 * T)     # avoid Omega -> 0 (trivial identity)
    bounds = (
        [(omega_lo, omega_max)]        # omega
        + [(0.0, 2.0 * np.pi)] * (N - 1)  # phi_2 ... phi_N
    )

    def unpack(x):
        omega  = float(x[0])
        phases = np.concatenate([[0.0], x[1:]])   # prepend phi_1 = 0
        return omega, phases

    def objective(x):
        omega, phases = unpack(x)
        try:
            seq = build_equiangular_3level(system, T, N, omega, phases)
            _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)
            if sens_sq < 1e-20:
                return 1e10
            freqs, Fe, _, _ = fft_three_level_filter(
                seq, n_samples=n_fft, pad_factor=pad_factor, m_y=m_y)
            kubo_var = float(simpson(Fe * S_func(freqs), x=freqs) / (2.0 * np.pi))
            if kubo_var < 1e-30:
                return 1e10
            return kubo_var / sens_sq
        except Exception:
            return 1e10

    # ── Global search ────────────────────────────────────────────────────────
    de_result = differential_evolution(
        objective,
        bounds,
        seed=int(rng.integers(2**31)),
        popsize=popsize,
        tol=tol,
        maxiter=maxiter,
        polish=True,
        workers=1,
    )
    best_x   = de_result.x.copy()
    best_val = de_result.fun

    # ── Local polishing from random restarts ─────────────────────────────────
    for _ in range(n_restarts):
        x0 = np.array(
            [rng.uniform(omega_lo, omega_max)]
            + list(rng.uniform(0.0, 2.0 * np.pi, N - 1))
        )
        res = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x   = res.x.copy()

    # ── Final evaluation ─────────────────────────────────────────────────────
    omega_opt, phases_opt = unpack(best_x)
    seq_opt = build_equiangular_3level(system, T, N, omega_opt, phases_opt)
    sens_sq, kubo_var, fom = evaluate_sequence(
        seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
        n_fft=n_fft, pad_factor=pad_factor)

    return PulseOptimizationResult(
        omega=omega_opt,
        phases=phases_opt,
        sensitivity_sq=sens_sq,
        kubo_var=kubo_var,
        fom=fom,
        noise_label=noise_label,
        seq=seq_opt,
    )


# =============================================================================
# Convenience: sweep over N
# =============================================================================

def sweep_n_segments(
    system,
    T: float,
    N_values,
    noise_psd: Union[str, Callable] = 'white',
    **kwargs,
):
    """
    Run optimize_equiangular_sequence for each N in N_values.

    Returns a list of PulseOptimizationResult, one per N.
    Extra keyword arguments are forwarded to optimize_equiangular_sequence.
    """
    return [
        optimize_equiangular_sequence(system, T, N, noise_psd=noise_psd, **kwargs)
        for N in N_values
    ]
