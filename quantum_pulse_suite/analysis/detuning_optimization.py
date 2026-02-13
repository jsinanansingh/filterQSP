"""
Detuning optimization for pulse sequence frequency estimation.

Provides tools to find the optimal detuning for each protocol by minimizing
the frequency estimation variance:

    σ²_freq = ⟨δ(B·σ)²⟩ / |∂⟨B·σ⟩/∂δ|²

When B_lab=None (default), the measurement direction is automatically
chosen as the Bloch-vector gradient dr/dδ, yielding the Cramér-Rao-optimal
single-shot variance.
"""

import numpy as np
from collections import namedtuple
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar


OptimizationResult = namedtuple('OptimizationResult', ['delta_opt', 'min_variance'])


def _rotation_matrix(f, g):
    """SO(3) rotation matrix from SU(2) Cayley-Klein parameters."""
    a, b = np.real(f), np.imag(f)
    c, d = np.real(g), np.imag(g)
    return np.array([
        [a**2 + c**2 - b**2 - d**2, 2*(c*d - a*b), 2*(a*d + b*c)],
        [2*(a*b + c*d), a**2 + d**2 - b**2 - c**2, 2*(b*d - a*c)],
        [2*(b*c - a*d), 2*(a*c + b*d), a**2 + b**2 - c**2 - d**2]
    ])


def _bloch_vector(seq):
    """Final Bloch vector for initial state |↑⟩."""
    U = seq.total_unitary()
    R = _rotation_matrix(U[0, 0], U[0, 1] / 1j)
    return R @ np.array([0.0, 0.0, 1.0])


def _filter_function_data(seq):
    """Return (Fx, Fy, Fz, R) for a sequence — reusable building blocks."""
    ff = seq.get_filter_function_calculator()
    U = seq.total_unitary()
    R = _rotation_matrix(U[0, 0], U[0, 1] / 1j)
    return ff, R


def compute_signal(seq, B_lab):
    """
    Compute ⟨B_lab · σ⟩ for initial state |↑⟩.

    Parameters
    ----------
    seq : PulseSequence
        A pulse sequence with polynomials already computed.
    B_lab : array_like, shape (3,)
        Lab-frame measurement axis (e.g., [0, 1, 0] for σ_y).

    Returns
    -------
    float
        The expectation value ⟨B_lab · σ⟩.
    """
    B_lab = np.asarray(B_lab, dtype=float)
    return float(B_lab @ _bloch_vector(seq))


def compute_noise_variance(seq, frequencies, S_func, B_lab):
    """
    Compute noise variance ⟨δ(B_lab·σ)²⟩ via the Kubo formula.

        (2/π) ∫ S(ω) [|F(ω)|² - |B̃·F(ω)|²] dω

    where B̃ = R^T @ B_lab is B_lab rotated into the toggling frame.

    Parameters
    ----------
    seq : PulseSequence
        A pulse sequence with polynomials already computed.
    frequencies : array_like
        Angular frequencies for integration.
    S_func : callable
        Noise power spectral density S(ω).
    B_lab : array_like, shape (3,)
        Lab-frame measurement axis.

    Returns
    -------
    float
        The noise variance.
    """
    B_lab = np.asarray(B_lab, dtype=float)
    ff = seq.get_filter_function_calculator()
    Fx, Fy, Fz = ff.filter_function(frequencies)
    U = seq.total_unitary()
    R = _rotation_matrix(U[0, 0], U[0, 1] / 1j)
    B = R.T @ B_lab
    F_sq = np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2
    BdotF = B[0]*Fx + B[1]*Fy + B[2]*Fz
    sens = np.maximum(F_sq - np.abs(BdotF)**2, 0.0)
    return float(2 * simpson(S_func(frequencies) * sens, x=frequencies) / np.pi)


def compute_signal_slope(seq_builder, delta, B_lab, eps=1e-6):
    """
    Numerical central-difference ∂⟨B_lab·σ⟩/∂δ.

    Parameters
    ----------
    seq_builder : callable
        delta → PulseSequence (with polynomials computed).
    delta : float
        Detuning at which to evaluate the slope.
    B_lab : array_like, shape (3,)
        Lab-frame measurement axis.
    eps : float
        Step size for central difference.

    Returns
    -------
    float
        The signal slope.
    """
    s_plus = compute_signal(seq_builder(delta + eps), B_lab)
    s_minus = compute_signal(seq_builder(delta - eps), B_lab)
    return (s_plus - s_minus) / (2 * eps)


def _bloch_gradient(seq_builder, delta, eps=1e-6):
    """Compute dr/dδ via central difference on Bloch vectors."""
    r_plus = _bloch_vector(seq_builder(delta + eps))
    r_minus = _bloch_vector(seq_builder(delta - eps))
    return (r_plus - r_minus) / (2 * eps)


def frequency_estimation_variance(seq_builder, delta, frequencies, S_func,
                                  B_lab=None, eps=1e-6):
    """
    Compute frequency estimation variance σ²_freq = ⟨δσ²⟩ / |∂⟨σ⟩/∂δ|².

    When B_lab is None, automatically selects the measurement direction
    that minimizes σ²_freq (the Cramér-Rao optimal direction dr/dδ).

    Parameters
    ----------
    seq_builder : callable
        delta → PulseSequence (with polynomials computed).
    delta : float
        Detuning.
    frequencies : array_like
        Angular frequencies for Kubo integration.
    S_func : callable
        Noise PSD.
    B_lab : array_like, shape (3,) or None
        Lab-frame measurement axis. If None, uses the optimal direction.
    eps : float
        Step size for slope computation.

    Returns
    -------
    float
        Frequency estimation variance. Returns inf when slope ≈ 0.
    """
    if B_lab is None:
        # Optimal measurement direction: parallel to dr/dδ
        dr = _bloch_gradient(seq_builder, delta, eps)
        slope_mag = np.linalg.norm(dr)
        if slope_mag < 1e-12:
            return np.inf
        B_opt = dr / slope_mag
        seq = seq_builder(delta)
        noise_var = compute_noise_variance(seq, frequencies, S_func, B_opt)
        return noise_var / slope_mag**2
    else:
        seq = seq_builder(delta)
        noise_var = compute_noise_variance(seq, frequencies, S_func, B_lab)
        slope = compute_signal_slope(seq_builder, delta, B_lab, eps)
        if abs(slope) < 1e-12:
            return np.inf
        return noise_var / slope**2


def optimize_detuning(seq_builder, frequencies, S_func, B_lab=None,
                      delta_range=(0.01, 5.0), n_grid=200):
    """
    Find the detuning that minimizes frequency estimation variance.

    1. Coarse grid scan over delta_range (n_grid points)
    2. Evaluate frequency estimation variance at each grid point
    3. Refine best point with scipy.optimize.minimize_scalar (bounded)

    When B_lab is None (default), uses the Cramér-Rao optimal measurement
    direction at each grid point (the Bloch-vector gradient dr/dδ).

    Parameters
    ----------
    seq_builder : callable
        delta → PulseSequence (with polynomials computed).
    frequencies : array_like
        Angular frequencies for Kubo integration.
    S_func : callable
        Noise PSD.
    B_lab : array_like, shape (3,) or None
        Lab-frame measurement axis. If None, auto-optimizes direction.
    delta_range : tuple of (float, float)
        (min_delta, max_delta) search range.
    n_grid : int
        Number of coarse grid points.

    Returns
    -------
    OptimizationResult
        Named tuple with (delta_opt, min_variance).
    """
    frequencies = np.asarray(frequencies)
    deltas = np.linspace(delta_range[0], delta_range[1], n_grid)

    if B_lab is not None:
        B_lab = np.asarray(B_lab, dtype=float)

    if B_lab is None:
        # Auto-optimal direction: build sequences once, compute Bloch vectors
        # and filter functions, then find optimal direction at each grid point.
        seqs = []
        bloch_vecs = np.zeros((n_grid, 3))
        for i, d in enumerate(deltas):
            seq = seq_builder(d)
            seqs.append(seq)
            bloch_vecs[i] = _bloch_vector(seq)

        # Slopes from finite differences on the Bloch vector grid
        dr_ddelta = np.gradient(bloch_vecs, deltas, axis=0)  # (n_grid, 3)
        slope_mags = np.linalg.norm(dr_ddelta, axis=1)       # (n_grid,)

        # Noise variance in the optimal direction at each grid point
        noise_vars = np.full(n_grid, np.inf)
        for i in range(n_grid):
            if slope_mags[i] < 1e-12:
                continue
            B_opt = dr_ddelta[i] / slope_mags[i]
            noise_vars[i] = compute_noise_variance(
                seqs[i], frequencies, S_func, B_opt)

        with np.errstate(divide='ignore', invalid='ignore'):
            variances = np.where(
                slope_mags > 1e-12,
                noise_vars / slope_mags**2,
                np.inf
            )
    else:
        # Fixed B_lab: compute signals and noise variances
        signals = np.zeros(n_grid)
        noise_vars = np.zeros(n_grid)
        for i, d in enumerate(deltas):
            seq = seq_builder(d)
            signals[i] = compute_signal(seq, B_lab)
            noise_vars[i] = compute_noise_variance(
                seq, frequencies, S_func, B_lab)

        slopes = np.gradient(signals, deltas)
        with np.errstate(divide='ignore', invalid='ignore'):
            variances = np.where(
                np.abs(slopes) > 1e-12,
                noise_vars / slopes**2,
                np.inf
            )

    # Best grid point
    best_idx = int(np.argmin(variances))
    best_var = variances[best_idx]
    best_delta = deltas[best_idx]

    if np.isinf(best_var):
        return OptimizationResult(best_delta, best_var)

    # Refine with bounded scalar minimization
    spacing = deltas[1] - deltas[0]
    bracket_lo = max(delta_range[0], best_delta - 2 * spacing)
    bracket_hi = min(delta_range[1], best_delta + 2 * spacing)

    try:
        result = minimize_scalar(
            lambda d: frequency_estimation_variance(
                seq_builder, d, frequencies, S_func, B_lab),
            bounds=(bracket_lo, bracket_hi),
            method='bounded',
            options={'xatol': 1e-6, 'maxiter': 30}
        )
        if result.fun < best_var:
            return OptimizationResult(result.x, result.fun)
    except Exception:
        pass

    return OptimizationResult(best_delta, best_var)
