"""
Optimised pulse-shaped GPS: sweep over operating detuning delta (Step 1)
for fixed envelope shapes, then optimise the Fourier envelope shape at
delta=0 (Step 2).

Envelope parameterisation
--------------------------
Omega(t) = Omega_mean * [ 1 - sum_{k=1}^{N} b_k cos(2*pi*k*t/T) ]

Constraints enforced analytically:
  * integral_0^T Omega(t) dt = Omega_mean * T = 2*pi*m   (cos terms -> 0)
  * Omega(0) = Omega(T) = 0  =>  sum_k b_k = 1,
    enforced by setting b_1 = 1 - sum_{k>=2} b_k.

Free parameters for Step 2: (b_2, ..., b_{N_FREE+1}) with delta fixed at 0.
Non-negativity Omega(t)>=0 is penalised in the objective.

All sequences are discretised into N_DISC constant-Omega segments.
Filter function: fft_three_level_filter (handles delta in each segment).
Sensitivity:     detuning_sensitivity(seq)  [delta baked into segments].

Objective (FOM)
---------------
  minimize  1/sens_sq + noise_var / noise_var_ramsey

where noise_var_ramsey is the Ramsey noise variance for the same noise PSD.
Integration lower cutoff: omega_min = 2*pi/T  (Fourier limit).
The objective_mode label 'ramsey_normalized' is stored in the cache
so results from different FOM choices can be identified.

Noise types optimised
---------------------
  white      : S(w) = 1
  high-pass1 : S(w) = theta(|w| - 1)   (omega_c = 1 rad/s)
  high-pass2 : S(w) = theta(|w| - 2)   (omega_c = 2 rad/s)

GPS cases
---------
  m=1 : Omega_mean = 1  rad/s,  delta_max = 3
  m=2 : Omega_mean = 2  rad/s,  delta_max = 6
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson
from scipy.optimize import differential_evolution, minimize, minimize_scalar

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
from quantum_pulse_suite.core.three_level_filter import (
    fft_three_level_filter, analytic_filter, detuning_sensitivity,
    default_omega_cutoff, gps_shaped_filter,
)
from quantum_pulse_suite.analysis.pulse_optimizer import (
    white_noise_psd, one_over_f_psd, high_pass_psd,
)

# =============================================================================
# Global config
# =============================================================================

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi
N_DISC     = 64          # segments for shaped sequence (optimisation)
N_DISC_FIN = 256         # segments for final evaluation
N_FFT      = 1024        # fft samples during optimisation
N_FFT_FIN  = 4096        # fft samples for final evaluation
PAD        = 4
FREQS_PLOT = np.logspace(-1, np.log10(30), 600)
N_FREE     = 4           # free Fourier coefficients b_2..b_{N_FREE+1}
OMEGA_CUTOFF = 2 * np.pi / T   # Fourier limit: 2pi/T = 1.0 rad/s

# Dense log-spaced grid for smooth analytic plot curves
FREQS_ANA = np.logspace(-1, np.log10(35), 1500)

OBJECTIVE_MODE = 'ramsey_normalized'

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'
CACHE_DIR  = OUTPUT_DIR / 'cache'

# Noise PSD specs: (cache_key, display_label, S_func)
NOISE_SPECS = [
    ('white', 'White',              white_noise_psd()),
    ('hp1',   'High-pass (wc=1)',  high_pass_psd(omega_c=1.0)),
    ('hp2',   'High-pass (wc=2)',  high_pass_psd(omega_c=2.0)),
]

# Deterministic per-noise seed offsets
_NOISE_SEED_OFFSET = {'white': 0, 'hp1': 5, 'hp2': 13}


# =============================================================================
# Cache helpers
# =============================================================================

def _latest_cache(required_key='objective_mode'):
    """Return path to the most recent cache containing required_key, or None."""
    if not CACHE_DIR.exists():
        return None
    candidates = sorted(CACHE_DIR.glob('shaped_gps_opt_cache_*.npz'), reverse=True)
    for p in candidates:
        try:
            c = np.load(str(p), allow_pickle=True)
            if required_key in c:
                return p
        except Exception:
            pass
    return None


def _save_cache(data: dict):
    """Save data dict to a timestamped npz in CACHE_DIR."""
    from datetime import datetime
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = CACHE_DIR / f'shaped_gps_opt_cache_{ts}.npz'
    np.savez(str(path), **data)
    return path


# =============================================================================
# Envelope functions
# =============================================================================

def envelope_square(ts, T_total, omega_mean):
    return np.full_like(ts, omega_mean)

def envelope_hann(ts, T_total, omega_mean):
    return omega_mean * (1.0 - np.cos(2.0 * np.pi * ts / T_total))

def envelope_blackman(ts, T_total, omega_mean):
    a0, a1, a2 = 0.42, 0.50, 0.08
    return (omega_mean / a0) * (a0
                                - a1 * np.cos(2.0 * np.pi * ts / T_total)
                                + a2 * np.cos(4.0 * np.pi * ts / T_total))

def make_cosine_envelope(b_free):
    """Return an envelope callable from free coefficients b_2..b_{N_FREE+1}."""
    b1    = 1.0 - float(np.sum(b_free))
    b_all = np.concatenate([[b1], np.asarray(b_free)])
    ks    = np.arange(1, len(b_all) + 1)

    def fn(ts, T_total, omega_mean):
        phases = 2.0 * np.pi * np.outer(ks, ts) / T_total  # (N, n_pts)
        return omega_mean * (1.0 - b_all @ np.cos(phases))
    return fn


def _planck_window(ts, T_total, eps):
    """Unnormalized Planck-taper weight: 1 on (eps*T, (1-eps)*T), C^inf ramp to 0 at ends."""
    t  = np.asarray(ts, dtype=float)
    w  = np.zeros_like(t)
    t1 = eps * T_total

    # Left ramp: t in (0, eps*T)
    mask_L = (t > 0) & (t < t1)
    if np.any(mask_L):
        u = t[mask_L] / t1                            # u in (0, 1)
        z = np.clip(1.0 / u - 1.0 / (1.0 - u), -500, 500)
        w[mask_L] = 1.0 / (1.0 + np.exp(z))

    # Flat center: t in [eps*T, (1-eps)*T]
    w[(t >= t1) & (t <= T_total - t1)] = 1.0

    # Right ramp: t in ((1-eps)*T, T)
    mask_R = (t > T_total - t1) & (t < T_total)
    if np.any(mask_R):
        u = (T_total - t[mask_R]) / t1
        z = np.clip(1.0 / u - 1.0 / (1.0 - u), -500, 500)
        w[mask_R] = 1.0 / (1.0 + np.exp(z))

    return w


def envelope_planck_taper(ts, T_total, omega_mean, eps=0.2):
    """Flat-top C^inf (Planck-taper) envelope.

    Flat at omega_mean on the central (1-2*eps) fraction of [0, T], with a
    C^inf smooth ramp to zero at each end over a fraction eps.  The amplitude
    is scaled so that int_0^T Omega(t) dt = omega_mean * T_total (same total
    rotation as the square envelope).
    """
    w = _planck_window(np.asarray(ts, dtype=float), T_total, eps)
    # Normalise: compute integral over fine grid then rescale
    t_quad  = np.linspace(0, T_total, 50001)
    w_quad  = _planck_window(t_quad, T_total, eps)
    integral = float(np.trapezoid(w_quad, t_quad))
    return omega_mean * T_total / integral * w


# =============================================================================
# Sequence builder
# =============================================================================

def build_seq(system, omega_mean, envelope_fn, delta, n_disc=N_DISC):
    tau    = T / n_disc
    ts_mid = (np.arange(n_disc) + 0.5) * tau
    omegas = np.maximum(envelope_fn(ts_mid, T, omega_mean), 1e-9)
    seq    = MultiLevelPulseSequence(system, system.probe)
    for ok in omegas:
        seq.add_continuous_pulse(ok, [1, 0, 0], delta, tau)
    seq.compute_polynomials()
    return seq


def build_ramsey(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


# =============================================================================
# FOM and negativity penalty
# =============================================================================

def compute_fom(seq, S_func, sens_ref_ramsey, noise_ref_ramsey, n_fft=N_FFT, pad=PAD):
    """Return (fom, sens_sq, noise_var, (freqs, Fe)).

    fom = sens_ref_ramsey/sens_sq + noise_var/noise_ref_ramsey   [ramsey_normalized]
    Both terms equal 1.0 at Ramsey, total = 2.0 at Ramsey.
    """
    _, sens_sq = detuning_sensitivity(seq)
    if sens_sq < 1e-20:
        return 1e10, 0.0, 0.0, None
    freqs, Fe, _, _ = fft_three_level_filter(seq, n_samples=n_fft, pad_factor=pad, m_y=1.0)
    mask = freqs >= OMEGA_CUTOFF
    if np.count_nonzero(mask) < 2:
        return 1e10, sens_sq, 0.0, (freqs, Fe)
    noise_var = float(simpson(Fe[mask] * S_func(freqs[mask]), x=freqs[mask]) / (2 * np.pi))
    fom = max(sens_ref_ramsey, 1e-15) / sens_sq + noise_var / max(noise_ref_ramsey, 1e-15)
    return fom, sens_sq, noise_var, (freqs, Fe)


def negativity_penalty(envelope_fn, omega_mean, n_check=256):
    """Penalty proportional to the integral of negative Omega(t)."""
    ts  = (np.arange(n_check) + 0.5) * T / n_check
    om  = envelope_fn(ts, T, omega_mean)
    neg = np.minimum(om, 0.0)
    return float(-np.sum(neg) * T / n_check)   # >= 0


# =============================================================================
# Step 1: optimise delta for a fixed envelope (per noise type)
# =============================================================================

def optimise_delta(system, omega_mean, envelope_fn, delta_max,
                   S_func, sens_ref_ramsey, noise_ref_ramsey,
                   n_scan=60, label=''):
    """Scan then polish to find the best delta for a fixed envelope shape."""
    deltas = np.linspace(-delta_max, delta_max, n_scan)
    best_fom, best_d = np.inf, 0.0

    for d in deltas:
        pen = negativity_penalty(envelope_fn, omega_mean)
        if pen > 1e-3:
            continue
        seq = build_seq(system, omega_mean, envelope_fn, d)
        fom, _, _, _ = compute_fom(seq, S_func, sens_ref_ramsey, noise_ref_ramsey)
        if fom < best_fom:
            best_fom, best_d = fom, d

    def fom_obj(d):
        seq = build_seq(system, omega_mean, envelope_fn, float(d))
        fom, _, _, _ = compute_fom(seq, S_func, sens_ref_ramsey, noise_ref_ramsey)
        return fom

    res = minimize_scalar(fom_obj,
                          bounds=(best_d - delta_max / 5, best_d + delta_max / 5),
                          method='bounded',
                          options={'xatol': 1e-4})
    best_d   = float(res.x)
    best_fom = float(res.fun)
    if label:
        print(f'    {label}: delta*={best_d:.4f}  fom={best_fom:.4f}')
    return best_d, best_fom


# =============================================================================
# Step 2: optimise envelope (Fourier coefficients) + delta jointly
# =============================================================================

def optimise_envelope(system, omega_mean, delta_max, S_func, sens_ref_ramsey, noise_ref_ramsey,
                      seed=42, label='', popsize=12, maxiter=250, n_restarts=3):
    """Joint optimisation over free Fourier coefficients b_2..b_{N_FREE+1} and delta."""
    b_bounds = [(-1.5, 1.5)] * N_FREE
    d_bound  = [(-delta_max, delta_max)]
    bounds   = b_bounds + d_bound

    def objective(x):
        b_free = x[:N_FREE]
        delta  = float(x[N_FREE])
        fn     = make_cosine_envelope(b_free)
        pen    = negativity_penalty(fn, omega_mean) * 1e4
        seq    = build_seq(system, omega_mean, fn, delta)
        fom, _, _, _ = compute_fom(seq, S_func, sens_ref_ramsey, noise_ref_ramsey)
        return fom + pen

    rng = np.random.default_rng(seed)
    de  = differential_evolution(
        objective, bounds,
        seed=int(rng.integers(2**31)),
        popsize=popsize, maxiter=maxiter,
        tol=1e-5, polish=False, workers=1,
    )
    best_x, best_val = de.x.copy(), de.fun

    for _ in range(n_restarts):
        x0 = rng.uniform([lo for lo, _ in bounds], [hi for _, hi in bounds])
        r  = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-7})
        if r.success and r.fun < best_val:
            best_val, best_x = r.fun, r.x.copy()

    b_opt  = best_x[:N_FREE]
    d_opt  = float(best_x[N_FREE])
    fn_opt = make_cosine_envelope(b_opt)
    b1_opt = 1.0 - float(np.sum(b_opt))
    if label:
        b_all = np.concatenate([[b1_opt], b_opt])
        print(f'    {label}: delta*={d_opt:.4f}  fom={best_val:.4f}')
        print(f'      b = [{", ".join(f"{v:.4f}" for v in b_all)}]')
    return d_opt, best_val, fn_opt, b_opt


# =============================================================================
# Step 2b: optimise envelope at delta=0 fixed
# =============================================================================

def optimise_envelope_delta0(system, omega_mean, S_func, sens_ref_ramsey, noise_ref_ramsey,
                              seed=42, label='', popsize=12, maxiter=250, n_restarts=3):
    """Optimise free Fourier coefficients b_2..b_{N_FREE+1} with delta fixed at 0."""
    bounds = [(-1.5, 1.5)] * N_FREE

    def objective(b_free):
        fn  = make_cosine_envelope(b_free)
        pen = negativity_penalty(fn, omega_mean) * 1e4
        seq = build_seq(system, omega_mean, fn, 0.0)
        fom, _, _, _ = compute_fom(seq, S_func, sens_ref_ramsey, noise_ref_ramsey)
        return fom + pen

    rng = np.random.default_rng(seed)
    de  = differential_evolution(
        objective, bounds,
        seed=int(rng.integers(2**31)),
        popsize=popsize, maxiter=maxiter,
        tol=1e-5, polish=False, workers=1,
    )
    best_x, best_val = de.x.copy(), de.fun

    for _ in range(n_restarts):
        x0 = rng.uniform([-1.5] * N_FREE, [1.5] * N_FREE)
        r  = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-7})
        if r.success and r.fun < best_val:
            best_val, best_x = r.fun, r.x.copy()

    b_opt  = best_x
    fn_opt = make_cosine_envelope(b_opt)
    b1_opt = 1.0 - float(np.sum(b_opt))
    if label:
        b_all = np.concatenate([[b1_opt], b_opt])
        print(f'    {label}: fom={best_val:.4f}  (delta=0 fixed)')
        print(f'      b = [{", ".join(f"{v:.4f}" for v in b_all)}]')
    return fn_opt, b_opt, best_val


# =============================================================================
# Final high-resolution evaluation (all noise types)
# =============================================================================

def final_eval(system, omega_mean, envelope_fn, delta, label='', verify=False):
    """Evaluate a sequence at high resolution under all noise types.

    Uses fft_three_level_filter for numerically stable results (analytic_filter
    overflows for sequences with many segments).  For delta=0 smooth plot curves,
    also calls gps_shaped_filter(method='direct') when envelope_fn is given.
    """
    seq = build_seq(system, omega_mean, envelope_fn, delta, n_disc=N_DISC_FIN)
    _, sens_sq = detuning_sensitivity(seq)

    # FFT filter — numerically stable for 256-segment sequences
    freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
        seq, n_samples=N_FFT_FIN, pad_factor=4, m_y=1.0)
    mask_fft = freqs_fft >= OMEGA_CUTOFF

    result = dict(sens_sq=sens_sq, delta=delta)
    for nkey, _, S_func in NOISE_SPECS:
        if np.count_nonzero(mask_fft) < 2:
            noise_var = 0.0
        else:
            noise_var = float(simpson(Fe_fft[mask_fft] * S_func(freqs_fft[mask_fft]),
                                      x=freqs_fft[mask_fft]) / (2 * np.pi))
        sigma_nu = noise_var / sens_sq if (sens_sq > 1e-20 and noise_var > 0) else np.inf
        result[f'noise_var_{nkey}'] = noise_var
        result[f'sigma_nu_{nkey}']  = sigma_nu

    # For plotting: use gps_shaped_filter direct method at delta=0 for smooth curves;
    # otherwise fall back to the FFT data already computed.
    if delta == 0.0 and envelope_fn is not None:
        freqs_smooth, Fe_smooth = gps_shaped_filter(
            envelope_fn, T, omega_mean, method='direct',
            n_samples=16384, pad_factor=4)
        result['freqs'] = freqs_smooth
        result['Fe']    = Fe_smooth
    else:
        result['freqs'] = freqs_fft
        result['Fe']    = Fe_fft

    # Optional cross-check: compare FFT vs direct gps_shaped_filter at delta=0
    if verify and delta == 0.0:
        freqs_d, Fe_d = gps_shaped_filter(
            envelope_fn, T, omega_mean, method='direct',
            n_samples=16384, pad_factor=4)
        mask_d = freqs_d >= OMEGA_CUTOFF
        for nkey, _, S_func in NOISE_SPECS:
            nv_d   = float(simpson(Fe_d[mask_d] * S_func(freqs_d[mask_d]),
                                   x=freqs_d[mask_d]) / (2 * np.pi))
            nv_fft = result[f'noise_var_{nkey}']
            rel    = abs(nv_d - nv_fft) / max(abs(nv_fft), 1e-30)
            print(f'    [verify {nkey}] direct={nv_d:.6e}  FFT={nv_fft:.6e}  '
                  f'rel_diff={rel:.4f}')

    if label:
        parts = [f'  {label:<46}  sens={sens_sq:.4f}']
        for nkey, _, _ in NOISE_SPECS:
            parts.append(f'  snu_{nkey}={result[f"sigma_nu_{nkey}"]:.4f}')
        parts.append(f'  d={delta:.4f}')
        print(''.join(parts))
    return result


# =============================================================================
# Method verification: direct vs piecewise gps_shaped_filter
# =============================================================================

def verify_gps_methods(system, omega_mean, envelope_fn, m_label='',
                       n_direct=16384, n_piecewise=N_DISC_FIN, pad=PAD):
    """Compare gps_shaped_filter(method='direct') vs method='piecewise' for delta=0.

    'direct'    uses the commuting-H analytic formula on the continuous envelope.
    'piecewise' uses fft_three_level_filter on an N_DISC_FIN-segment sequence.

    Prints noise_var for each noise type and the relative difference between
    the two methods.  Also returns the relative differences as a dict.
    """
    # Direct method: continuous envelope, no piecewise approximation
    freqs_d, Fe_d = gps_shaped_filter(
        envelope_fn, T, omega_mean, method='direct',
        n_samples=n_direct, pad_factor=pad)

    # Piecewise method: build a high-res discretised sequence, delegate to FFT
    seq_pw = build_seq(system, omega_mean, envelope_fn, delta=0.0, n_disc=n_piecewise)
    freqs_p, Fe_p = gps_shaped_filter(
        envelope_fn, T, omega_mean, method='piecewise',
        n_samples=4 * n_piecewise, pad_factor=pad, seq=seq_pw)

    rel_diffs = {}
    print(f'  gps_shaped_filter method comparison  [{m_label}, delta=0]')
    hdr = f'    {"noise":<12}  {"direct":>14}  {"piecewise":>14}  {"rel_diff":>10}'
    print(hdr)
    for nkey, nlabel, S_func in NOISE_SPECS:
        mask_d = freqs_d >= OMEGA_CUTOFF
        mask_p = freqs_p >= OMEGA_CUTOFF
        nv_d = float(simpson(Fe_d[mask_d] * S_func(freqs_d[mask_d]),
                             x=freqs_d[mask_d]) / (2 * np.pi))
        nv_p = float(simpson(Fe_p[mask_p] * S_func(freqs_p[mask_p]),
                             x=freqs_p[mask_p]) / (2 * np.pi))
        rel  = abs(nv_d - nv_p) / max(abs(nv_d), 1e-30)
        rel_diffs[nkey] = rel
        print(f'    {nlabel:<12}  {nv_d:>14.6e}  {nv_p:>14.6e}  {rel:>10.4f}')
    return rel_diffs


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    gps_cases = [
        (1, 2 * np.pi * 1 / T, 3.0),    # (m, omega_mean, delta_max)
    ]
    fixed_shapes = [
        ('Square',   envelope_square),
        ('Hann',     envelope_hann),
        ('Blackman', envelope_blackman),
        ('Planck',   envelope_planck_taper),
    ]
    _all_names = [s for s, _ in fixed_shapes] + ['Opt0']

    # cache key lists used for save/load
    _nkeys = [nkey for nkey, _, _ in NOISE_SPECS]
    _per_noise_suffixes = (
        [f'noise_var_{k}' for k in _nkeys]
        + [f'sigma_nu_{k}'  for k in _nkeys]
    )
    _shared_suffixes = ('sens_sq', 'delta', 'freqs', 'Fe')

    # ── Try to load from cache ─────────────────────────────────────────────────
    _cache_path = _latest_cache(required_key='objective_mode')
    cache_ok    = False
    if _cache_path is not None:
        try:
            cache = np.load(str(_cache_path), allow_pickle=True)
            cached_mode   = cache['objective_mode'].item()
            cached_cutoff = float(cache['omega_cutoff']) if 'omega_cutoff' in cache else -1.0
            mode_ok   = cached_mode == OBJECTIVE_MODE
            cutoff_ok = abs(cached_cutoff - OMEGA_CUTOFF) < 1e-10
            keys_ok   = all(
                f'm{m}_{nk}_{nm}_sens_sq' in cache
                for m, _, _ in gps_cases
                for nk in [nkey for nkey, _, _ in NOISE_SPECS]
                for nm in [s for s, _ in fixed_shapes] + ['Opt0']
            )
            if mode_ok and cutoff_ok and keys_ok:
                cache_ok = True
                print(f'Loading cached results from {_cache_path} '
                      f'(mode={cached_mode}, cutoff={cached_cutoff:.2e})')
            else:
                print(f'Cache mismatch: mode={cached_mode}/{OBJECTIVE_MODE}, '
                      f'cutoff={cached_cutoff:.2e}/{OMEGA_CUTOFF:.2e}. Recomputing.')
        except Exception as e:
            print(f'Cache load failed ({e}). Recomputing.')

    if cache_ok:
        ramsey_sens_sq = float(cache['ramsey_sens_sq'])
        ramsey_freqs   = cache['ramsey_freqs']
        ramsey_Fe      = cache['ramsey_Fe']
        ramsey_refs    = {nkey: float(cache[f'ramsey_{nkey}_noise_var']) for nkey in _nkeys}

        results    = {m: {nk: {} for nk in _nkeys} for m, _, _ in gps_cases}
        opt_params = {m: {nk: {} for nk in _nkeys} for m, _, _ in gps_cases}
        for m, _, _ in gps_cases:
            for nkey in _nkeys:
                for name in _all_names:
                    key = f'm{m}_{nkey}_{name}'
                    r = {}
                    for k in _shared_suffixes:
                        v = cache[f'{key}_{k}']
                        r[k] = float(v) if v.ndim == 0 else v
                    for k in _per_noise_suffixes:
                        r[k] = float(cache[f'{key}_{k}'])
                    results[m][nkey][name] = r
                opt_params[m][nkey]['opt0_b'] = cache[f'm{m}_{nkey}_opt0_b']
        print('  Done.')

    else:
        # ── Ramsey baseline ───────────────────────────────────────────────────
        print('Computing Ramsey baseline ...')
        seq_r = build_ramsey(system)
        _, ramsey_sens_sq = detuning_sensitivity(seq_r)
        # Use analytic filter for accurate Kubo integrals and smooth plot data
        _, Fe_r_ana = analytic_filter(seq_r, FREQS_ANA, m_y=1.0)
        mask_r = FREQS_ANA >= OMEGA_CUTOFF

        ramsey_refs = {}
        for nkey, nlabel, S_func in NOISE_SPECS:
            nv = float(simpson(Fe_r_ana[mask_r] * S_func(FREQS_ANA[mask_r]),
                               x=FREQS_ANA[mask_r]) / (2 * np.pi))
            ramsey_refs[nkey] = nv
            sigma_r = nv / ramsey_sens_sq if (nv > 0 and ramsey_sens_sq > 1e-20) else np.inf
            print(f'  Ramsey [{nlabel}]: sens_sq={ramsey_sens_sq:.4f}  '
                  f'noise_var={nv:.4e}  sigma_nu={sigma_r:.4f}')
        ramsey_freqs = FREQS_ANA
        ramsey_Fe    = Fe_r_ana

        results    = {m: {nk: {} for nk in _nkeys} for m, _, _ in gps_cases}
        opt_params = {m: {nk: {} for nk in _nkeys} for m, _, _ in gps_cases}

        # ── Optimization loop: per noise type, per m, per shape ───────────────
        for nkey, nlabel, S_func in NOISE_SPECS:
            noise_ref = ramsey_refs[nkey]
            print(f'\n{"="*60}')
            print(f'Noise: {nlabel}  (noise_ref_ramsey={noise_ref:.4e})')
            print(f'{"="*60}')

            for m_cyc, omega_mean, delta_max in gps_cases:
                print(f'\n--- GPS m={m_cyc}  (Omega_mean={omega_mean:.4f}) ---')
                seed_base = 17 + m_cyc + _NOISE_SEED_OFFSET[nkey]

                # Step 1: fixed shapes
                print('  Step 1 – fixed shapes, optimise delta:')
                first_verify = True
                for name, fn in fixed_shapes:
                    d_opt, _ = optimise_delta(
                        system, omega_mean, fn, delta_max,
                        S_func, ramsey_sens_sq, noise_ref,
                        label=f'{name} m={m_cyc} [{nlabel}]')
                    results[m_cyc][nkey][name] = final_eval(
                        system, omega_mean, fn, d_opt,
                        label=f'GPS m={m_cyc}  {name} [{nlabel}]',
                        verify=first_verify)
                    first_verify = False

                # Step 2: optimise envelope at delta=0
                print('  Step 2b – optimise envelope at delta=0:')
                t0 = time.time()
                fn_opt0, b_opt0, _ = optimise_envelope_delta0(
                    system, omega_mean, S_func, ramsey_sens_sq, noise_ref,
                    seed=seed_base + 14,
                    label=f'Opt0 m={m_cyc} [{nlabel}]',
                    popsize=12, maxiter=250, n_restarts=4)
                print(f'    (took {time.time()-t0:.1f}s)')
                opt_params[m_cyc][nkey]['opt0_b'] = b_opt0
                results[m_cyc][nkey]['Opt0'] = final_eval(
                    system, omega_mean, fn_opt0, 0.0,
                    label=f'GPS m={m_cyc}  Opt0 [{nlabel}]')

        # ── Save cache ────────────────────────────────────────────────────────
        d = {
            'objective_mode':  np.array(OBJECTIVE_MODE),
            'omega_cutoff':    np.array(OMEGA_CUTOFF),
            'ramsey_sens_sq':  np.array(ramsey_sens_sq),
            'ramsey_freqs':    ramsey_freqs,
            'ramsey_Fe':       ramsey_Fe,
        }
        for nkey in _nkeys:
            d[f'ramsey_{nkey}_noise_var'] = np.array(ramsey_refs[nkey])
        for m_cyc, _, _ in gps_cases:
            for nkey in _nkeys:
                for name in _all_names:
                    key = f'm{m_cyc}_{nkey}_{name}'
                    r   = results[m_cyc][nkey][name]
                    for k in _shared_suffixes:
                        d[f'{key}_{k}'] = np.array(r[k])
                    for k in _per_noise_suffixes:
                        d[f'{key}_{k}'] = np.array(r[k])
                d[f'm{m_cyc}_{nkey}_opt0_b'] = opt_params[m_cyc][nkey]['opt0_b']
        saved = _save_cache(d)
        print(f'\nCache saved to {saved}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n\n=== Summary (sigma_nu per noise type) ===')
    col_w = 16
    hdr = f'{"Protocol":<48}' + ''.join(f'{nlbl:>{col_w}}' for _, nlbl, _ in NOISE_SPECS)
    print(hdr);  print('-' * len(hdr))

    ramsey_row = f'  {"Ramsey":<46}'
    for nkey in _nkeys:
        nv  = ramsey_refs[nkey]
        sig = nv / ramsey_sens_sq if (nv > 0 and ramsey_sens_sq > 1e-20) else np.inf
        ramsey_row += f'{sig:>{col_w}.4f}'
    print(ramsey_row)

    for m_cyc, _, _ in gps_cases:
        for name in _all_names:
            # Show white-noise optimal result's sigma_nu across all noise types
            r   = results[m_cyc]['white'][name]
            lbl = f'GPS m={m_cyc}  {name} [white-opt]'
            row = f'  {lbl:<46}'
            for nkey in _nkeys:
                row += f'{r[f"sigma_nu_{nkey}"]:>{col_w}.4f}'
            print(row)

    print('\n--- Per-noise-type optimal sequences ---')
    for nkey, nlabel, _ in NOISE_SPECS:
        print(f'\n  [{nlabel}]')
        for m_cyc, _, _ in gps_cases:
            for name in ['Opt0']:
                r   = results[m_cyc][nkey][name]
                lbl = f'GPS m={m_cyc}  {name}'
                row = f'    {lbl:<44}'
                for nk in _nkeys:
                    row += f'{r[f"sigma_nu_{nk}"]:>{col_w}.4f}'
                print(row + f'  delta={r["delta"]:.4f}')

    # ── Recompute filter functions at plotting resolution ─────────────────────
    # Use gps_shaped_filter(direct) for all GPS curves (delta=0, smooth).
    # pad_factor=32 -> df = 1/32 ~ 0.031 rad/s, so curves start below 0.05.
    # n_samples must be a power of 2 for FFT efficiency.
    m_cyc, omega_mean, _ = gps_cases[0]

    PAD_PLOT  = 32
    N_T_PLOT  = 8192   # time samples; df = 1/pad_factor = 0.031 rad/s

    _envelope_fns = {
        'Square':   envelope_square,
        'Hann':     envelope_hann,
        'Blackman': envelope_blackman,
        'Planck':   envelope_planck_taper,
    }

    plot_freqs, plot_Fe = {}, {}
    for name in _all_names:
        if name == 'Opt0':
            b   = opt_params[m_cyc]['white']['opt0_b']
            fn  = make_cosine_envelope(b)
        else:
            fn = _envelope_fns[name]
        f, Fe_p = gps_shaped_filter(
            fn, T, omega_mean, method='direct',
            n_samples=N_T_PLOT, pad_factor=PAD_PLOT)
        plot_freqs[name] = f
        plot_Fe[name]    = Fe_p

    # Ramsey: analytic filter on log-spaced grid for smooth log-log curve
    FREQS_RAMSEY_PLOT = np.logspace(np.log10(0.04), np.log10(32), 3000)
    seq_r_plot = build_ramsey(system)
    _, Fe_ramsey_plot = analytic_filter(seq_r_plot, FREQS_RAMSEY_PLOT, m_y=1.0)

    # ── PRA-standard plot style ───────────────────────────────────────────────
    matplotlib.rcParams.update({
        'font.family':       'serif',
        'font.size':          8,
        'axes.labelsize':     9,
        'axes.titlesize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'legend.framealpha': 0.9,
        'legend.edgecolor':  '0.7',
        'lines.linewidth':    1.4,
        'axes.linewidth':     0.6,
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'xtick.minor.width':  0.4,
        'ytick.minor.width':  0.4,
        'xtick.direction':   'in',
        'ytick.direction':   'in',
    })

    # ── Single figure: GPS m=1 filter functions (white noise) ────────────────
    fig, ax = plt.subplots(figsize=(3.375, 3.0))

    ax.loglog(FREQS_RAMSEY_PLOT, Fe_ramsey_plot / T**2 + 1e-20,
              color='0.5', lw=1.5, ls='--',
              label=rf'Ramsey  $\sigma_{{\nu}}={ramsey_refs["white"]/ramsey_sens_sq:.3f}$')

    color_map = {'Square': 'C0', 'Hann': 'C9', 'Blackman': 'C1', 'Planck': 'C4', 'Opt0': 'C3'}
    ls_map    = {'Square': '-',  'Hann': '--', 'Blackman': ':',  'Planck': '-.',  'Opt0': '-'}
    lw_map    = {'Square': 1.8,  'Hann': 2.0, 'Blackman': 2.0,  'Planck': 2.0,   'Opt0': 2.5}

    for name in _all_names:
        sigma = results[m_cyc]['white'][name]['sigma_nu_white']
        lbl   = rf'{name}  $\sigma_{{\nu}}={sigma:.3f}$'
        ax.loglog(plot_freqs[name], plot_Fe[name] / T**2 + 1e-20,
                  color=color_map[name], lw=lw_map[name],
                  ls=ls_map[name], label=lbl)

    # Asymptotic slope guides
    w_lo = np.array([2.5, 7.0])
    ax.loglog(w_lo, 2e-2 * (w_lo / 2.5) ** (-4),  color='0.65', lw=0.9, ls=':', alpha=0.8)
    ax.loglog(w_lo, 2e-2 * (w_lo / 2.5) ** (-12), color='0.65', lw=0.9, ls=':', alpha=0.8)
    ax.text(4.2, 2e-2 * (4.2 / 2.5) ** (-4)  * 2.2, r'$\omega^{-4}$',  fontsize=8, color='0.5')
    ax.text(3.8, 2e-2 * (3.8 / 2.5) ** (-12) * 0.3, r'$\omega^{-12}$', fontsize=8, color='0.5')

    # Inset: envelope shapes
    ax_ins = ax.inset_axes([0.12, 0.35, 0.28, 0.28])
    ts = np.linspace(0, T, 500)
    ax_ins.plot(ts / np.pi, envelope_square(ts, T, omega_mean) / omega_mean,
                color='C0', lw=1.2, ls='-',  label='Square')
    ax_ins.plot(ts / np.pi, envelope_hann(ts, T, omega_mean) / omega_mean,
                color='C9', lw=1.5, ls='--', label='Hann')
    ax_ins.plot(ts / np.pi, envelope_blackman(ts, T, omega_mean) / omega_mean,
                color='C1', lw=1.5, ls=':',  label='Blackman')
    ax_ins.plot(ts / np.pi, envelope_planck_taper(ts, T, omega_mean) / omega_mean,
                color='C4', lw=1.5, ls='-.', label='Planck')
    b_opt0  = opt_params[m_cyc]['white']['opt0_b']
    fn_opt0 = make_cosine_envelope(b_opt0)
    ax_ins.plot(ts / np.pi, fn_opt0(ts, T, omega_mean) / omega_mean,
                color='C3', lw=2.0, ls='-',  label='Opt0')
    ax_ins.set_xlabel(r'$t/\pi$')
    ax_ins.set_ylabel(r'$\Omega/\bar\Omega$')
    ax_ins.tick_params(which='both', labelsize=6)
    ax_ins.set_xlim(0, 2)
    ax_ins.set_ylim(bottom=0)
    ax_ins.grid(True, alpha=0.3)

    ax.set_ylim([1e-9, 1.5])
    ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'$\mathcal{F}_e(\omega)\,/\,T^2$')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(which='both', top=True, right=True)
    ax.legend(loc='lower left')
    fig.tight_layout()
    x_min = float(FREQS_RAMSEY_PLOT[0])
    ax.set_xlim([x_min, 30])

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'shaped_gps_filter_functions.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
