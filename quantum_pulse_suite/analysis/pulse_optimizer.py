"""
Pulse sequence optimizer for equiangular continuous-drive sequences.

Searches for the N-segment equiangular pulse sequence that minimises the
normalised frequency estimation variance

    sigma_nu = noise_var / sens_sq
             = [int dw/2pi S(w) F(w)] / F(0)

where the classical noise filter function is F(w) = m_y^2 * |Chi(w)|^2
(Chi = FT[|G(t)|^2]) and F(0) = sens_sq (signal slope squared).

Parameterisation
----------------
* N continuous-drive segments of equal duration tau = T / N  (equiangular)
* Each segment is driven at a shared Rabi frequency Omega with drive axis
  (cos(phi_k), sin(phi_k), 0) in the xy-plane.
* Free parameters: Omega > 0 and phases phi_2, ..., phi_N  (phi_1 = 0 by
  gauge choice, reducing the redundancy from global phase freedom).

The objective minimised is noise_var / sensitivity_sq  (= sigma_nu).

Noise PSDs
----------
* white   : S(omega) = S0  (constant)
* 1/f     : S(omega) = S0 / |omega|  (1/f noise; integrated from the lowest
            FFT frequency, well above any 1/f corner expected in practice)

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
    analytic_filter,
    detuning_sensitivity,
    resolve_omega_cutoff,
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


def mixed_noise_psd(
    omega_c1: float,
    omega_c2: float,
    lo_cutoff: float = 1e-4,
) -> Callable:
    """
    Two-corner noise PSD for a realistic oscillator.

    S(omega) = (1 + omega_c1 / |omega|) * theta(omega_c2 - |omega|)

    Regions
    -------
    |omega| << omega_c1       :  S ~ omega_c1 / |omega|  (1/f dominated)
    omega_c1 << |omega| << omega_c2  :  S ~ 1  (white noise floor)
    |omega| > omega_c2        :  S = 0  (high-frequency cutoff)

    Parameters
    ----------
    omega_c1 : float
        1/f corner frequency (rad/s).  Set to 0 to recover pure white noise
        below omega_c2.
    omega_c2 : float
        High-frequency cutoff (rad/s).  Set to np.inf for no cutoff.
    lo_cutoff : float
        Small regularisation to avoid 1/|omega| divergence at omega=0;
        should be well below the lowest FFT frequency.
    """
    def S(w):
        w   = np.asarray(w, dtype=float)
        s   = 1.0 + omega_c1 / np.maximum(np.abs(w), lo_cutoff)
        if np.isfinite(omega_c2):
            s = s * (np.abs(w) <= omega_c2)
        return s
    return S


def high_pass_psd(omega_c: float) -> Callable:
    """
    High-pass white noise: S(omega) = 1 for |omega| >= omega_c, else 0.

    Models a noise floor that is negligible below omega_c (e.g. the LO
    white-noise floor above the 1/f corner).
    """
    def S(w):
        return (np.abs(np.asarray(w, dtype=float)) >= omega_c).astype(float)
    return S


_PSD_PRESETS = {
    'white':      white_noise_psd,
    '1/f':        one_over_f_psd,
    'high_pass':  high_pass_psd,   # requires omega_c kwarg; use callable directly
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
    omega_cutoff: Optional[float] = None,
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
        F(omega=0) = |chi(0)|^2 = |partial_delta <M>|^2 at delta=0.
    noise_var : float
        Bandlimited noise variance:
        integral_{omega_cutoff}^inf F(omega) S(omega) domega / (2 pi).
        The default lower cutoff is the Fourier limit 2*pi/T; set
        ``omega_cutoff=0.0`` to integrate from DC.
    sigma_nu : float
        Normalised frequency variance noise_var / sensitivity_sq.
    """
    _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)

    T = seq.total_duration()
    omega_lo = resolve_omega_cutoff(T, omega_cutoff)
    freqs, Fe, _, _ = fft_three_level_filter(
        seq, n_samples=n_fft, pad_factor=pad_factor, m_y=m_y)
    mask = freqs >= omega_lo
    if np.count_nonzero(mask) < 2:
        noise_var = 0.0
    else:
        noise_var = float(simpson(Fe[mask] * S_func(freqs[mask]), x=freqs[mask]) / (2.0 * np.pi))

    if sens_sq > 0.0 and noise_var > 0.0:
        sigma_nu = noise_var / sens_sq
    else:
        sigma_nu = 0.0

    return sens_sq, noise_var, sigma_nu


def analytic_evaluate_sequence(
    seq: MultiLevelPulseSequence,
    S_func: Callable,
    m_y: float = 1.0,
    M=None,
    psi0=None,
    n_omega: int = 512,
    omega_max: float = 50.0,
    fft_pad_factor: int = 4,
    omega_cutoff: Optional[float] = None,
):
    """
    Evaluate a 3-level probe sequence using analytic filter integrals.

    Replaces the FFT Kubo integral with an analytic evaluation on a uniform
    frequency grid, which is significantly faster for equiangular sequences
    because ``analytic_filter`` is fully vectorised over frequencies and
    requires only O(N_segments) Python iterations, whereas the FFT path loops
    over O(n_fft) time samples in Python.

    The default lower frequency limit is 2*pi/T (the Fourier limit: lowest
    frequency resolvable in a measurement of duration T).  Set
    ``omega_cutoff=0.0`` to integrate from DC.

    Parameters
    ----------
    seq : MultiLevelPulseSequence
    S_func : callable
        Noise PSD S(omega).
    m_y : float
        Measurement weight.
    M, psi0 : optional
        Passed through to detuning_sensitivity.
    n_omega : int
        Number of frequency quadrature nodes (default 512).
    omega_max : float
        Upper limit of frequency grid in rad/s (default 50.0).
        Should comfortably exceed the highest significant Fe peak.
    fft_pad_factor : int
        Unused, kept for API compatibility.

    Returns
    -------
    sensitivity_sq, noise_var, sigma_nu : float
    """
    _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)

    T = seq.total_duration()
    omega_lo = resolve_omega_cutoff(T, omega_cutoff)
    freqs = np.linspace(omega_lo, omega_max, n_omega)
    if omega_lo >= omega_max:
        noise_var = 0.0
    else:
        _, Fe = analytic_filter(seq, freqs, m_y=m_y)
        noise_var = float(simpson(Fe * S_func(freqs), x=freqs) / (2.0 * np.pi))

    if sens_sq > 0.0 and noise_var > 0.0:
        sigma_nu = noise_var / sens_sq
    else:
        sigma_nu = 0.0

    return sens_sq, noise_var, sigma_nu


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
        |partial_delta <M>|^2 at the optimum.  Equals F(0).
    noise_var : float
        Classical noise variance: integral F(omega) S(omega) domega / (2 pi).
    sigma_nu : float
        Normalised frequency variance noise_var / sensitivity_sq.
    noise_label : str
        Label of the noise PSD used.
    seq : MultiLevelPulseSequence
        The optimised sequence.
    """
    omega: float
    phases: np.ndarray
    sensitivity_sq: float
    noise_var: float
    sigma_nu: float
    noise_label: str
    seq: object


# =============================================================================
# QSP sequence: n instantaneous pulses with equal free-evolution gaps
# =============================================================================

@dataclass
class QSPOptimizationResult:
    """
    Result of a QSP pulse-sequence optimisation.

    Attributes
    ----------
    n : int
        Number of pulses.
    thetas : np.ndarray, shape (n,)
        Optimal rotation angles theta_j in (0, 2*pi].
    phis : np.ndarray, shape (n,)
        Optimal rotation phases phi_j; phi[0] = 0 by convention.
    omega_fast : float
        Fixed fast Rabi frequency used for pulses (rad/s).
    tau_free : float
        Free-evolution gap duration (s).
    sensitivity_sq : float
        |partial_delta <M>|^2 at the optimum.  Equals F(0).
    noise_var : float
        Classical noise variance: integral F(omega) S(omega) domega / (2 pi).
    sigma_nu : float
        Normalised frequency variance noise_var / sensitivity_sq.
    noise_label : str
    seq : MultiLevelPulseSequence
    """
    n: int
    thetas: np.ndarray
    phis: np.ndarray
    omega_fast: float
    tau_free: float
    sensitivity_sq: float
    noise_var: float
    sigma_nu: float
    noise_label: str
    seq: object


def build_qsp_3level(
    system,
    T: float,
    n: int,
    thetas,
    phis,
    omega_fast: float,
) -> MultiLevelPulseSequence:
    """
    Build an n-pulse QSP sequence on the three-level Lambda clock.

    Protocol
    --------
    R(theta_1, phi_1) − free(tau_free) − R(theta_2, phi_2) − ... − R(theta_n, phi_n)

    Each pulse R(theta_j, phi_j) is a fast rotation by angle theta_j about
    axis (cos phi_j, sin phi_j, 0) at Rabi frequency omega_fast, lasting
    tau_j = theta_j / omega_fast.  Free-evolution gaps are all equal:

        tau_free = (T - sum_j theta_j / omega_fast) / (n - 1)

    so the total time is exactly T.

    Parameters
    ----------
    system : ThreeLevelClock
    T : float
        Total interrogation time (rad/s units, same as rest of codebase).
    n : int
        Number of pulses.
    thetas : array_like, shape (n,)
        Pulse rotation angles (rad); must satisfy sum(thetas)/omega_fast < T.
    phis : array_like, shape (n,)
        Pulse phase angles (rad); phis[0] = 0 by convention.
    omega_fast : float
        Rabi frequency for pulses (rad/s).
    """
    thetas = np.asarray(thetas, dtype=float)
    phis   = np.asarray(phis,   dtype=float)

    tau_pulse_total = float(np.sum(thetas)) / omega_fast
    tau_free = (T - tau_pulse_total) / max(n - 1, 1)
    if tau_free < 0.0:
        raise ValueError(
            f"Pulse durations sum to {tau_pulse_total:.4f} > T={T:.4f}")

    seq = MultiLevelPulseSequence(system, system.probe)
    for j in range(n):
        tau_j = float(thetas[j]) / omega_fast
        nhat  = [np.cos(float(phis[j])), np.sin(float(phis[j])), 0.0]
        seq.add_continuous_pulse(omega_fast, nhat, 0.0, tau_j)
        if j < n - 1:
            seq.add_free_evolution(tau_free, 0.0)

    seq.compute_polynomials()
    return seq


def optimize_qsp_sequence(
    system,
    T: float,
    n: int,
    noise_psd: Union[str, Callable] = 'white',
    omega_fast: float = 20.0 * np.pi,
    m_y: float = 1.0,
    M=None,
    psi0=None,
    n_restarts: int = 5,
    seed: int = 0,
    n_fft: int = 2048,
    pad_factor: int = 4,
    omega_cutoff: Optional[float] = None,
    popsize: int = 15,
    tol: float = 1e-5,
    maxiter: int = 300,
) -> QSPOptimizationResult:
    """
    Optimise an n-pulse QSP sequence on the three-level Lambda clock.

    Free parameters: theta_1 ... theta_n in (0, pi],  phi_2 ... phi_n in [0, 2*pi).
    phi_1 = 0 by gauge choice.  The upper bound theta <= pi restricts each
    pulse to at most a pi-rotation (standard QSP convention); the optimiser
    can find any rotation in (0, pi] on each pulse.

    Returns
    -------
    QSPOptimizationResult
    """
    rng = np.random.default_rng(seed)

    if isinstance(noise_psd, str):
        if noise_psd not in _PSD_PRESETS:
            raise ValueError(f"noise_psd must be 'white', '1/f', or callable; got {noise_psd!r}")
        S_func = _PSD_PRESETS[noise_psd]()
        noise_label = noise_psd
    else:
        S_func = noise_psd
        noise_label = 'custom'

    # Parameter layout: x = [theta_1,...,theta_n, phi_2,...,phi_n]  (2n-1 params)
    theta_lo  = 0.01
    theta_hi  = np.pi          # at most a pi-pulse per element
    phi_lo    = 0.0
    phi_hi    = 2.0 * np.pi
    bounds = [(theta_lo, theta_hi)] * n + [(phi_lo, phi_hi)] * (n - 1)

    def unpack(x):
        thetas = x[:n]
        phis   = np.concatenate([[0.0], x[n:]])
        return thetas, phis

    def objective(x):
        thetas, phis = unpack(x)
        try:
            tau_free = (T - float(np.sum(thetas)) / omega_fast) / max(n - 1, 1)
            if tau_free < 1e-6:
                return 1e10
            seq = build_qsp_3level(system, T, n, thetas, phis, omega_fast)
            _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)
            if sens_sq < 1e-20:
                return 1e10
            freqs, Fe, _, _ = fft_three_level_filter(
                seq, n_samples=n_fft, pad_factor=pad_factor, m_y=m_y)
            omega_lo = resolve_omega_cutoff(T, omega_cutoff)
            mask = freqs >= omega_lo
            if np.count_nonzero(mask) < 2:
                return 1e10
            noise_var = float(simpson(Fe[mask] * S_func(freqs[mask]), x=freqs[mask]) / (2.0 * np.pi))
            if noise_var < 1e-30:
                return 1e10
            return noise_var / sens_sq
        except Exception:
            return 1e10

    # Global search
    de_result = differential_evolution(
        objective, bounds,
        seed=int(rng.integers(2**31)),
        popsize=popsize, tol=tol, maxiter=maxiter,
        polish=True, workers=1,
    )
    best_x   = de_result.x.copy()
    best_val = de_result.fun

    # Local polishing
    for _ in range(n_restarts):
        x0 = np.concatenate([
            rng.uniform(theta_lo, theta_hi, n),
            rng.uniform(phi_lo,   phi_hi,   n - 1),
        ])
        res = minimize(
            objective, x0,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x   = res.x.copy()

    thetas_opt, phis_opt = unpack(best_x)
    seq_opt  = build_qsp_3level(system, T, n, thetas_opt, phis_opt, omega_fast)
    tau_free = (T - float(np.sum(thetas_opt)) / omega_fast) / max(n - 1, 1)
    sens_sq, noise_var, sigma_nu = evaluate_sequence(
        seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
        n_fft=n_fft, pad_factor=pad_factor, omega_cutoff=omega_cutoff)

    return QSPOptimizationResult(
        n=n,
        thetas=thetas_opt,
        phis=phis_opt,
        omega_fast=omega_fast,
        tau_free=tau_free,
        sensitivity_sq=sens_sq,
        noise_var=noise_var,
        sigma_nu=sigma_nu,
        noise_label=noise_label,
        seq=seq_opt,
    )


# =============================================================================
# Optimiser (equiangular)
# =============================================================================

def optimize_equiangular_sequence(
    system,
    T: float,
    N: int,
    noise_psd: Union[str, Callable] = 'white',
    omega_max: Optional[float] = None,
    omega_fixed: Optional[float] = None,
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
    use_analytic: bool = True,
    n_omega: int = 512,
    omega_max_analytic: float = 50.0,
    omega_cutoff: Optional[float] = None,
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
        Ignored when omega_fixed is set.
    omega_fixed : float, optional
        If provided, fix Omega to this value and optimise phases only.
        Useful for isolating the effect of phase modulation vs GPS at the
        same Rabi frequency.
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
        FFT sample count for filter function (default 2048).  Only used when
        use_analytic=False.
    pad_factor : int
        FFT zero-padding factor (default 4).  Only used when use_analytic=False.
    popsize : int
        DE population multiplier (default 15).
    tol : float
        DE convergence tolerance (default 1e-5).
    maxiter : int
        DE maximum generations (default 300).
    use_analytic : bool
        If True (default), use analytic_filter for the Kubo integral in the
        optimiser objective.  This is substantially faster than the FFT path
        because analytic_filter is vectorised over frequencies and requires
        only O(N) Python iterations, whereas the FFT path loops over O(n_fft)
        time samples in Python.
    n_omega : int
        Number of quadrature nodes for the analytic Kubo integral (default 512).
        Only used when use_analytic=True.
    omega_max_analytic : float
        Upper frequency limit for the analytic Kubo integral (default 50.0 rad/s).
        Should comfortably exceed the highest significant Fe peak of the sequence.
        Only used when use_analytic=True.

    Returns
    -------
    PulseOptimizationResult
    """
    rng = np.random.default_rng(seed)

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

    # Precompute analytic frequency grid (reused across all objective evaluations).
    _omega_lo = resolve_omega_cutoff(T, omega_cutoff)
    if use_analytic:
        _ana_freqs = np.linspace(_omega_lo, omega_max_analytic, n_omega)

    def _kubo(seq):
        """Compute noise variance from omega_lo = 2*pi/T to omega_max."""
        if use_analytic:
            if _omega_lo >= omega_max_analytic:
                return 0.0
            _, Fe = analytic_filter(seq, _ana_freqs, m_y=m_y)
            return float(simpson(Fe * S_func(_ana_freqs), x=_ana_freqs) / (2.0 * np.pi))
        else:
            freqs, Fe, _, _ = fft_three_level_filter(
                seq, n_samples=n_fft, pad_factor=pad_factor, m_y=m_y)
            mask = freqs >= _omega_lo
            if np.count_nonzero(mask) < 2:
                return 0.0
            return float(simpson(Fe[mask] * S_func(freqs[mask]), x=freqs[mask]) / (2.0 * np.pi))

    if omega_fixed is not None:
        # ── Phase-only optimisation at fixed Omega ────────────────────────────
        # x = [phi_2, ..., phi_N]  (N-1 free phases; phi_1 = 0 fixed)
        phase_bounds = [(0.0, 2.0 * np.pi)] * (N - 1)

        def unpack_phases(x):
            phases = np.concatenate([[0.0], x])
            return omega_fixed, phases

        def objective_phases(x):
            omega, phases = unpack_phases(x)
            try:
                seq = build_equiangular_3level(system, T, N, omega, phases)
                _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)
                if sens_sq < 1.0:
                    return 1e10
                kubo_var = _kubo(seq)
                if kubo_var < 1e-30:
                    return 1e10
                return kubo_var / sens_sq
            except Exception:
                return 1e10

        if N == 1:
            # No free phases: single segment at fixed omega, evaluate directly.
            best_phases = np.array([0.0])
            best_omega  = omega_fixed
        else:
            de_result = differential_evolution(
                objective_phases,
                phase_bounds,
                seed=int(rng.integers(2**31)),
                popsize=popsize,
                tol=tol,
                maxiter=maxiter,
                polish=True,
                workers=1,
            )
            best_x_ph  = de_result.x.copy()
            best_val   = de_result.fun

            for _ in range(n_restarts):
                x0 = rng.uniform(0.0, 2.0 * np.pi, N - 1)
                res = minimize(
                    objective_phases, x0,
                    method='L-BFGS-B',
                    bounds=phase_bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
                )
                if res.success and res.fun < best_val:
                    best_val  = res.fun
                    best_x_ph = res.x.copy()

            best_omega, best_phases = unpack_phases(best_x_ph)

        seq_opt = build_equiangular_3level(system, T, N, best_omega, best_phases)
        if use_analytic:
            sens_sq, noise_var, sigma_nu = analytic_evaluate_sequence(
                seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
                n_omega=n_omega, omega_max=omega_max_analytic,
                omega_cutoff=omega_cutoff)
        else:
            sens_sq, noise_var, sigma_nu = evaluate_sequence(
                seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
                n_fft=n_fft, pad_factor=pad_factor,
                omega_cutoff=omega_cutoff)
        return PulseOptimizationResult(
            omega=best_omega,
            phases=best_phases,
            sensitivity_sq=sens_sq,
            noise_var=noise_var,
            sigma_nu=sigma_nu,
            noise_label=noise_label,
            seq=seq_opt,
        )

    # ── Joint Omega + phase optimisation ─────────────────────────────────────
    if omega_max is None:
        omega_max = N * np.pi / T   # maximum: one pi-pulse per segment

    # Parameter layout: x = [omega, phi_2, ..., phi_N]   (N free params total)
    # phi_1 = 0 is fixed to break the global-phase gauge redundancy.
    omega_lo = np.pi / (10.0 * T)    # avoid Omega -> 0 (trivial identity)
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
            kubo_var = _kubo(seq)
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
    if use_analytic:
        sens_sq, noise_var, sigma_nu = analytic_evaluate_sequence(
            seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
            n_omega=n_omega, omega_max=omega_max_analytic,
            omega_cutoff=omega_cutoff)
    else:
        sens_sq, noise_var, sigma_nu = evaluate_sequence(
            seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
            n_fft=n_fft, pad_factor=pad_factor,
            omega_cutoff=omega_cutoff)

    return PulseOptimizationResult(
        omega=omega_opt,
        phases=phases_opt,
        sensitivity_sq=sens_sq,
        noise_var=noise_var,
        sigma_nu=sigma_nu,
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
