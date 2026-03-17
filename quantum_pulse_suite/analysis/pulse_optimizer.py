"""
Pulse sequence optimizer for three-level Lambda clock sequences.

Searches for the pulse sequence that optimises a frequency-estimation figure
of merit built from the DC sensitivity and the off-DC noise variance.

    sigma_nu = noise_var / sens_sq
             = [int dw/2pi S(w) F(w)] / F(0)

where the probe-noise filter function is F(w) = |H(w)|^2 with
    H(w) = Re[F(T)] * Chi(w) + G(T)/2 * Phi(w) + G(T)*/2 * Phi(-w)*
(for sequences with G(T)=0, this reduces to F(w) = m_y^2 |Chi(w)|^2).
F(0) = sens_sq (signal slope squared).

Filter function implementations
--------------------------------
* ``analytic_filter``     — closed-form integrals over segment; vectorised over
                            frequencies.  Used by default in the optimiser for speed.
* ``fft_three_level_filter`` — direct matrix-multiplication ground-truth check.
                               Computes U(t), A_e(t) = U†(t)|e><e|U(t),
                               r(t) = <psi0|[M_I(T), A_e(t)]|psi0>, then FFTs r(t).
                               Should NOT be used in the optimiser inner loop.

Parameterisations supported
----------------------------
* Equiangular: N continuous-drive segments of equal duration tau = T/N, with
  shared Rabi frequency Omega and phases phi_1,...,phi_N.
* QSP: n instantaneous pulses (fast Rabi frequency omega_fast) with free
  rotation angles theta_j and phases phi_j, separated by equal free-evolution
  gaps.

Noise PSDs
----------
* white   : S(omega) = S0  (constant)
* 1/f     : S(omega) = S0 / |omega|  (1/f noise; integrated from the lowest
            frequency 2*pi/T, well above any 1/f corner expected in practice)

Algorithm
---------
1. Global search with scipy differential_evolution.
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


def build_ramsey_3level(system, T: float, omega_fast: float) -> MultiLevelPulseSequence:
    """Build a Ramsey reference sequence with two finite pi/2 pulses."""
    tau_pi2 = np.pi / (2.0 * omega_fast)
    tau_free = T - 2.0 * tau_pi2
    if tau_free <= 0.0:
        raise ValueError(f"Ramsey reference requires T > pi/omega_fast; got T={T:.4f}")
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega_fast, [1.0, 0.0, 0.0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(omega_fast, [1.0, 0.0, 0.0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


def _objective_from_metrics(
    sens_sq: float,
    noise_var: float,
    objective_mode: str,
    sens_ref: float,
    noise_ref: float,
    objective_weight: float,
) -> float:
    """Return a minimisation objective from sequence metrics.

    Modes
    -----
    'sigma_nu'
        Minimise noise_var / sens_sq  (normalised frequency variance).
    'normalized_difference'
        Maximise sens_sq/sens_ref - weight * noise_var/noise_ref  (returned
        negated so the optimiser minimises).
    'inv_sens_plus_noise'
        Minimise  1/sens_sq + noise_var/sens_sq^2  =  (1 + sigma_nu)/sens_sq.
        Both terms have the same units (1/sens_sq).  Positive-definite and
        simultaneously rewards high sensitivity and low noise variance.
    'inv_sens_plus_ramsey_noise'
        Minimise  1/sens_sq + noise_var/noise_ref.
        noise_ref is the Ramsey noise variance for the same noise PSD.
        Normalises the noise penalty by a fixed (protocol-independent) scale,
        so the two terms are independently interpretable.
    'ramsey_normalized'
        Minimise  sens_ref/sens_sq + noise_var/noise_ref.
        Both terms are dimensionless and equal 1.0 at the Ramsey reference,
        giving a total of 2.0 at Ramsey.  Any improvement in sensitivity OR
        noise reduces the objective below 2.  The two terms are on the same
        scale by construction.
    """
    if sens_sq <= 0.0 or not np.isfinite(sens_sq):
        return 1e10
    if noise_var < 0.0 or not np.isfinite(noise_var):
        return 1e10

    if objective_mode == 'sigma_nu':
        if noise_var <= 0.0:
            return 1e10
        return noise_var / sens_sq

    if objective_mode == 'normalized_difference':
        sens_scale = max(sens_ref, 1e-15)
        noise_scale = max(noise_ref, 1e-15)
        score = sens_sq / sens_scale - objective_weight * noise_var / noise_scale
        return -float(score)

    if objective_mode == 'inv_sens_plus_noise':
        return 1.0 / sens_sq + noise_var / sens_sq ** 2

    if objective_mode == 'inv_sens_plus_ramsey_noise':
        return 1.0 / sens_sq + noise_var / max(noise_ref, 1e-15)

    if objective_mode == 'ramsey_normalized':
        return max(sens_ref, 1e-15) / sens_sq + noise_var / max(noise_ref, 1e-15)

    raise ValueError(
        f"objective_mode must be 'sigma_nu', 'normalized_difference', "
        f"'inv_sens_plus_noise', 'inv_sens_plus_ramsey_noise', or "
        f"'ramsey_normalized'; got {objective_mode!r}"
    )


def _report_score(
    objective_mode: str,
    sens_sq: float,
    noise_var: float,
    sigma_nu: float,
    sens_ref: float,
    noise_ref: float,
    objective_weight: float,
) -> float:
    """Return the objective score for the result dataclass.

    For minimisation modes ('sigma_nu', 'inv_sens_plus_noise') this equals
    the raw objective value (lower is better).
    For 'normalized_difference' the stored score is negated so that higher
    is still better for human inspection.
    """
    raw = _objective_from_metrics(
        sens_sq, noise_var, objective_mode, sens_ref, noise_ref, objective_weight)
    return -raw if objective_mode == 'normalized_difference' else raw


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
        Optimal drive phases phi_k (rad).
    sensitivity_sq : float
        |partial_delta <M>|^2 at the optimum.  Equals F(0).
    noise_var : float
        Classical noise variance: integral F(omega) S(omega) domega / (2 pi).
    sigma_nu : float
        Normalised frequency variance noise_var / sensitivity_sq.
    objective_score : float
        Optimiser score for the selected objective convention.
    objective_mode : str
        Optimiser objective convention.
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
    objective_score: float = 0.0
    objective_mode: str = 'sigma_nu'


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
        Optimal rotation phases phi_j.
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
    objective_score : float
        Optimiser score for the selected objective convention.
    objective_mode : str
        Optimiser objective convention.
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
    objective_score: float = 0.0
    objective_mode: str = 'sigma_nu'


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
    objective_mode: str = 'normalized_difference',
    objective_weight: float = 1.0,
    popsize: int = 15,
    tol: float = 1e-5,
    maxiter: int = 300,
    use_analytic: bool = True,
    n_omega: int = 512,
    omega_max_analytic: float = 50.0,
) -> QSPOptimizationResult:
    """
    Optimise an n-pulse QSP sequence on the three-level Lambda clock.

    Free parameters: theta_1 ... theta_n in (0, pi],  phi_1 ... phi_n in [0, 2*pi).
    The upper bound theta <= pi restricts each
    pulse to at most a pi-rotation (standard QSP convention); the optimiser
    can find any rotation in (0, pi] on each pulse.

    use_analytic : bool
        If True (default), use analytic_filter for the noise integral in the
        objective.  Substantially faster than the FFT path for sequences with
        many short pulses.
    n_omega : int
        Quadrature nodes for the analytic integral (default 512).
    omega_max_analytic : float
        Upper frequency limit for the analytic integral (default 50.0 rad/s).

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

    ref_seq = build_ramsey_3level(system, T, omega_fast)
    _omega_lo = resolve_omega_cutoff(T, omega_cutoff)
    if use_analytic:
        _ana_freqs = np.linspace(_omega_lo, omega_max_analytic, n_omega)
        sens_ref, noise_ref, sigma_ref = analytic_evaluate_sequence(
            ref_seq, S_func, m_y=m_y, M=M, psi0=psi0,
            n_omega=n_omega, omega_max=omega_max_analytic,
            omega_cutoff=omega_cutoff)
    else:
        sens_ref, noise_ref, sigma_ref = evaluate_sequence(
            ref_seq, S_func, m_y=m_y, M=M, psi0=psi0,
            n_fft=n_fft, pad_factor=pad_factor, omega_cutoff=omega_cutoff)

    # Parameter layout: x = [theta_1,...,theta_n, phi_1,...,phi_n]  (2n params)
    theta_lo  = 0.01
    theta_hi  = np.pi          # at most a pi-pulse per element
    phi_lo    = 0.0
    phi_hi    = 2.0 * np.pi
    bounds = [(theta_lo, theta_hi)] * n + [(phi_lo, phi_hi)] * n

    def unpack(x):
        thetas = x[:n]
        phis   = x[n:]
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
            if use_analytic:
                if _omega_lo >= omega_max_analytic:
                    return 1e10
                _, Fe = analytic_filter(seq, _ana_freqs, m_y=m_y)
                noise_var = float(simpson(Fe * S_func(_ana_freqs), x=_ana_freqs) / (2.0 * np.pi))
            else:
                freqs, Fe, _, _ = fft_three_level_filter(
                    seq, n_samples=n_fft, pad_factor=pad_factor, m_y=m_y)
                mask = freqs >= _omega_lo
                if np.count_nonzero(mask) < 2:
                    return 1e10
                noise_var = float(simpson(Fe[mask] * S_func(freqs[mask]), x=freqs[mask]) / (2.0 * np.pi))
            return _objective_from_metrics(
                sens_sq, noise_var, objective_mode,
                sens_ref, noise_ref, objective_weight)
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
            rng.uniform(phi_lo,   phi_hi,   n),
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
    if use_analytic:
        sens_sq, noise_var, sigma_nu = analytic_evaluate_sequence(
            seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
            n_omega=n_omega, omega_max=omega_max_analytic,
            omega_cutoff=omega_cutoff)
    else:
        sens_sq, noise_var, sigma_nu = evaluate_sequence(
            seq_opt, S_func, m_y=m_y, M=M, psi0=psi0,
            n_fft=n_fft, pad_factor=pad_factor, omega_cutoff=omega_cutoff)
    objective_score = _report_score(
        objective_mode, sens_sq, noise_var, sigma_nu,
        sens_ref, noise_ref, objective_weight)

    return QSPOptimizationResult(
        n=n,
        thetas=thetas_opt,
        phis=phis_opt,
        omega_fast=omega_fast,
        tau_free=tau_free,
        sensitivity_sq=sens_sq,
        noise_var=noise_var,
        sigma_nu=sigma_nu,
        objective_score=objective_score,
        objective_mode=objective_mode,
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
    ramsey_omega_fast: float = 20.0 * np.pi,
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
    objective_mode: str = 'normalized_difference',
    objective_weight: float = 1.0,
) -> PulseOptimizationResult:
    """
    Optimise an equiangular continuous-drive 3-level clock sequence.

    Minimises the frequency estimation variance
        sigma^2_freq = Kubo_variance(S) / |partial_delta <M>|^2
    over N equal-duration continuous-drive segments, parametrised by a
    shared Rabi frequency Omega and N free drive phases.

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
    ramsey_omega_fast : float
        Rabi frequency used to build the Ramsey reference sequence for sens_ref
        and noise_ref.  Default: 20*pi (the standard fast-pulse Ramsey used by
        the QSP optimiser).  Using a consistent value across all optimisers
        ensures fair comparison under FOMs that depend on noise_ref.
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

    ref_seq = build_ramsey_3level(system, T, ramsey_omega_fast)
    if use_analytic:
        sens_ref, noise_ref, sigma_ref = analytic_evaluate_sequence(
            ref_seq, S_func, m_y=m_y, M=M, psi0=psi0,
            n_omega=n_omega, omega_max=omega_max_analytic,
            omega_cutoff=omega_cutoff)
    else:
        sens_ref, noise_ref, sigma_ref = evaluate_sequence(
            ref_seq, S_func, m_y=m_y, M=M, psi0=psi0,
            n_fft=n_fft, pad_factor=pad_factor,
            omega_cutoff=omega_cutoff)

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
        # x = [phi_1, ..., phi_N]
        phase_bounds = [(0.0, 2.0 * np.pi)] * N

        def unpack_phases(x):
            return omega_fixed, np.asarray(x, dtype=float)

        def objective_phases(x):
            omega, phases = unpack_phases(x)
            try:
                seq = build_equiangular_3level(system, T, N, omega, phases)
                _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)
                kubo_var = _kubo(seq)
                return _objective_from_metrics(
                    sens_sq, kubo_var, objective_mode,
                    sens_ref, noise_ref, objective_weight)
            except Exception:
                return 1e10

        if N == 1:
            # Single segment at fixed omega: optimise the lone phase directly.
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
                x0 = rng.uniform(0.0, 2.0 * np.pi, N)
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
            objective_score=_report_score(
                objective_mode, sens_sq, noise_var, sigma_nu,
                sens_ref, noise_ref, objective_weight),
            objective_mode=objective_mode,
            noise_label=noise_label,
            seq=seq_opt,
        )

    # ── Joint Omega + phase optimisation ─────────────────────────────────────
    if omega_max is None:
        omega_max = N * np.pi / T   # maximum: one pi-pulse per segment

    # Parameter layout: x = [omega, phi_1, ..., phi_N]   (N+1 params total)
    omega_lo = np.pi / (10.0 * T)    # avoid Omega -> 0 (trivial identity)
    bounds = (
        [(omega_lo, omega_max)]        # omega
        + [(0.0, 2.0 * np.pi)] * N
    )

    def unpack(x):
        omega  = float(x[0])
        phases = np.asarray(x[1:], dtype=float)
        return omega, phases

    def objective(x):
        omega, phases = unpack(x)
        try:
            seq = build_equiangular_3level(system, T, N, omega, phases)
            _, sens_sq = detuning_sensitivity(seq, M=M, psi0=psi0)
            kubo_var = _kubo(seq)
            return _objective_from_metrics(
                sens_sq, kubo_var, objective_mode,
                sens_ref, noise_ref, objective_weight)
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
            + list(rng.uniform(0.0, 2.0 * np.pi, N))
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
        objective_score=_report_score(
            objective_mode, sens_sq, noise_var, sigma_nu,
            sens_ref, noise_ref, objective_weight),
        objective_mode=objective_mode,
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
