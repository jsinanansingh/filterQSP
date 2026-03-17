"""
Optimised pulse-shaped GPS: sweep over operating detuning delta (Step 1),
then optimise the Fourier envelope shape jointly with delta (Step 2).

Envelope parameterisation
--------------------------
Omega(t) = Omega_mean * [ 1 - sum_{k=1}^{N} b_k cos(2*pi*k*t/T) ]

Constraints enforced analytically:
  * integral_0^T Omega(t) dt = Omega_mean * T = 2*pi*m   (cos terms -> 0)
  * Omega(0) = Omega(T) = 0  =>  sum_k b_k = 1,
    enforced by setting b_1 = 1 - sum_{k>=2} b_k.

Free parameters for Step 2: (b_2, ..., b_{N_FREE+1}, delta).
Non-negativity Omega(t)>=0 is penalised in the objective.

All sequences are discretised into N_DISC constant-Omega segments.
Filter function: fft_three_level_filter (handles delta in each segment).
Sensitivity:     detuning_sensitivity(seq)  [delta baked into segments].

Objective (FOM)
---------------
  minimize  1/sens_sq + noise_var / noise_var_ramsey

where noise_var_ramsey is the Ramsey noise variance for the same noise PSD.
The objective_mode label 'inv_sens_plus_ramsey_noise' is stored in the cache
so results from different FOM choices can be identified.

Noise types optimised
---------------------
  white     : S(w) = 1
  1/f       : S(w) = 1/|w|
  high-pass : S(w) = theta(|w| - 2)   (omega_c = 2 rad/s)
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
    fft_three_level_filter, detuning_sensitivity,
    default_omega_cutoff,
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
FREQS_PLOT = np.logspace(-1.5, np.log10(30), 600)
N_FREE     = 4           # free Fourier coefficients b_2..b_{N_FREE+1}
OMEGA_CUTOFF = default_omega_cutoff(T)

OBJECTIVE_MODE = 'ramsey_normalized'

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'
CACHE_DIR  = OUTPUT_DIR / 'cache'

# Noise PSD specs: (cache_key, display_label, S_func)
NOISE_SPECS = [
    ('white', 'White',              white_noise_psd()),
    ('1f',    '1/f',               one_over_f_psd()),
    ('hp2',   'High-pass (wc=2)',  high_pass_psd(omega_c=2.0)),
]

# Deterministic per-noise seed offsets
_NOISE_SEED_OFFSET = {'white': 0, '1f': 7, 'hp2': 13}


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

def final_eval(system, omega_mean, envelope_fn, delta, label=''):
    """Evaluate a sequence at high resolution under all noise types."""
    seq = build_seq(system, omega_mean, envelope_fn, delta, n_disc=N_DISC_FIN)
    _, sens_sq = detuning_sensitivity(seq)
    freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
        seq, n_samples=N_FFT_FIN, pad_factor=4, m_y=1.0)
    mask = freqs_fft >= OMEGA_CUTOFF

    result = dict(sens_sq=sens_sq, delta=delta)
    for nkey, _, S_func in NOISE_SPECS:
        if np.count_nonzero(mask) < 2:
            noise_var = 0.0
        else:
            noise_var = float(simpson(Fe_fft[mask] * S_func(freqs_fft[mask]),
                                      x=freqs_fft[mask]) / (2 * np.pi))
        sigma_nu = noise_var / sens_sq if (sens_sq > 1e-20 and noise_var > 0) else np.inf
        result[f'noise_var_{nkey}'] = noise_var
        result[f'sigma_nu_{nkey}']  = sigma_nu

    mask_plot = freqs_fft <= 35.0
    result['freqs'] = freqs_fft[mask_plot]
    result['Fe']    = Fe_fft[mask_plot]

    if label:
        parts = [f'  {label:<46}  sens={sens_sq:.4f}']
        for nkey, _, _ in NOISE_SPECS:
            parts.append(f'  snu_{nkey}={result[f"sigma_nu_{nkey}"]:.4f}')
        parts.append(f'  d={delta:.4f}')
        print(''.join(parts))
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    gps_cases = [
        (1, 2 * np.pi * 1 / T, 3.0),    # (m, omega_mean, delta_max)
        (8, 2 * np.pi * 8 / T, 24.0),
    ]
    fixed_shapes = [
        ('Square',   envelope_square),
        ('Hann',     envelope_hann),
        ('Blackman', envelope_blackman),
    ]
    _all_names = [s for s, _ in fixed_shapes] + ['Opt', 'Opt0']

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
            cached_mode = cache['objective_mode'].item()
            if cached_mode == OBJECTIVE_MODE:
                cache_ok = True
                print(f'Loading cached results from {_cache_path} (mode={cached_mode})')
            else:
                print(f'Cache mode={cached_mode} != {OBJECTIVE_MODE}. Recomputing.')
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
                opt_params[m][nkey]['opt_b']    = cache[f'm{m}_{nkey}_opt_b']
                opt_params[m][nkey]['opt_delta'] = float(cache[f'm{m}_{nkey}_opt_delta'])
                opt_params[m][nkey]['opt0_b']    = cache[f'm{m}_{nkey}_opt0_b']
        print('  Done.')

    else:
        # ── Ramsey baseline ───────────────────────────────────────────────────
        print('Computing Ramsey baseline ...')
        seq_r = build_ramsey(system)
        _, ramsey_sens_sq = detuning_sensitivity(seq_r)
        freqs_r, Fe_r, _, _ = fft_three_level_filter(
            seq_r, n_samples=N_FFT_FIN, pad_factor=4, m_y=1.0)
        mask_r = freqs_r >= OMEGA_CUTOFF

        ramsey_refs = {}
        for nkey, nlabel, S_func in NOISE_SPECS:
            nv = float(simpson(Fe_r[mask_r] * S_func(freqs_r[mask_r]),
                               x=freqs_r[mask_r]) / (2 * np.pi))
            ramsey_refs[nkey] = nv
            sigma_r = nv / ramsey_sens_sq if (nv > 0 and ramsey_sens_sq > 1e-20) else np.inf
            print(f'  Ramsey [{nlabel}]: sens_sq={ramsey_sens_sq:.4f}  '
                  f'noise_var={nv:.4e}  sigma_nu={sigma_r:.4f}')
        mask_r_plot = freqs_r <= 35.0
        ramsey_freqs = freqs_r[mask_r_plot]
        ramsey_Fe    = Fe_r[mask_r_plot]

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
                for name, fn in fixed_shapes:
                    d_opt, _ = optimise_delta(
                        system, omega_mean, fn, delta_max,
                        S_func, ramsey_sens_sq, noise_ref,
                        label=f'{name} m={m_cyc} [{nlabel}]')
                    results[m_cyc][nkey][name] = final_eval(
                        system, omega_mean, fn, d_opt,
                        label=f'GPS m={m_cyc}  {name} [{nlabel}]')

                # Step 2: optimise envelope + delta
                print('  Step 2 – optimise envelope + delta:')
                t0 = time.time()
                d_opt, _, fn_opt, b_opt = optimise_envelope(
                    system, omega_mean, delta_max, S_func, ramsey_sens_sq, noise_ref,
                    seed=seed_base,
                    label=f'Opt m={m_cyc} [{nlabel}]',
                    popsize=12, maxiter=250, n_restarts=4)
                print(f'    (took {time.time()-t0:.1f}s)')
                opt_params[m_cyc][nkey]['opt_b']    = b_opt
                opt_params[m_cyc][nkey]['opt_delta'] = d_opt
                results[m_cyc][nkey]['Opt'] = final_eval(
                    system, omega_mean, fn_opt, d_opt,
                    label=f'GPS m={m_cyc}  Opt [{nlabel}]')

                # Step 2b: optimise envelope at delta=0
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
                d[f'm{m_cyc}_{nkey}_opt_b']    = opt_params[m_cyc][nkey]['opt_b']
                d[f'm{m_cyc}_{nkey}_opt_delta'] = np.array(opt_params[m_cyc][nkey]['opt_delta'])
                d[f'm{m_cyc}_{nkey}_opt0_b']   = opt_params[m_cyc][nkey]['opt0_b']
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
            for name in ['Opt', 'Opt0']:
                r   = results[m_cyc][nkey][name]
                lbl = f'GPS m={m_cyc}  {name}'
                row = f'    {lbl:<44}'
                for nk in _nkeys:
                    row += f'{r[f"sigma_nu_{nk}"]:>{col_w}.4f}'
                print(row + f'  delta={r["delta"]:.4f}')

    # ── Filter function plots (one per noise type, showing that noise's optimum) ──
    for nkey, nlabel, _ in NOISE_SPECS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

        color_map = {'Square': 'C0', 'Hann': 'C9', 'Blackman': 'C1',
                     'Opt': 'C3', 'Opt0': 'C5'}
        ls_map    = {'Square': '-',  'Hann': '--', 'Blackman': ':',
                     'Opt': '-',  'Opt0': '-'}
        lw_map    = {'Square': 1.8,  'Hann': 2.0, 'Blackman': 2.0,
                     'Opt': 2.5,  'Opt0': 2.5}

        for ax, (m_cyc, omega_mean, _) in zip(axes, gps_cases):
            ax.loglog(ramsey_freqs, ramsey_Fe / T**2 + 1e-20,
                      color='0.55', lw=1.5, label='Ramsey (ref)')

            for name in _all_names:
                r     = results[m_cyc][nkey][name]
                sigma = r[f'sigma_nu_{nkey}']
                d_str = r'$\delta{=}0$' if name == 'Opt0' else rf'$\delta^*$={r["delta"]:.2f}'
                lbl   = rf'{name}  {d_str}  $\sigma_{{\nu}}$={sigma:.3f}'
                ax.loglog(r['freqs'], r['Fe'] / T**2 + 1e-20,
                          color=color_map[name], lw=lw_map[name],
                          ls=ls_map[name], label=lbl)

            # Inset: envelope shapes (Opt and Opt0 for this noise type)
            ax_ins = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
            ts = np.linspace(0, T, 300)
            ax_ins.plot(ts / np.pi, envelope_square(ts, T, omega_mean) / omega_mean,
                        color='C0', lw=1.2, ls='-', label='Square')
            b_opt  = opt_params[m_cyc][nkey]['opt_b']
            b_opt0 = opt_params[m_cyc][nkey]['opt0_b']
            fn_opt  = make_cosine_envelope(b_opt)
            fn_opt0 = make_cosine_envelope(b_opt0)
            ax_ins.plot(ts / np.pi, fn_opt(ts, T, omega_mean)  / omega_mean,
                        color='C3', lw=2.0, ls='-', label='Opt')
            ax_ins.plot(ts / np.pi, fn_opt0(ts, T, omega_mean) / omega_mean,
                        color='C5', lw=2.0, ls='-', label=r'Opt$\delta{=}0$')
            ax_ins.set_xlabel(r'$t/\pi$', fontsize=7)
            ax_ins.set_ylabel(r'$\Omega/\bar\Omega$', fontsize=7)
            ax_ins.tick_params(labelsize=6)
            ax_ins.set_xlim(0, 2);  ax_ins.set_ylim(bottom=0)
            ax_ins.legend(fontsize=5.5, loc='upper center')
            ax_ins.grid(True, alpha=0.3)

            for k in range(1, 6):
                if omega_mean * k <= 30:
                    ax.axvline(omega_mean * k, color='0.7', lw=0.5, ls=':', alpha=0.5)

            ax.set_xlim([FREQS_PLOT[0], 30]);  ax.set_ylim([1e-9, 1.5])
            ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=11)
            ax.set_title(rf'GPS $m{{=}}{m_cyc}$  ($\bar\Omega={omega_mean:.1f}$)',
                         fontsize=11)
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=7.5, loc='lower left')

        axes[0].set_ylabel(r'$F_e(\omega)\,/\,T^2$', fontsize=11)
        fig.suptitle(
            rf'Pulse-shaped GPS: filter functions optimised for {nlabel} noise'
            '\n'
            r'FOM = $1/S^2 + \sigma^2_\nu/\sigma^2_{\nu,\rm Ramsey}$  '
            r'($N_{\rm free}=' + str(N_FREE) + r'$ cosine terms)',
            fontsize=10)
        fig.tight_layout()

        suffix = nkey.replace('/', 'f')
        for ext in ['pdf', 'png']:
            path = OUTPUT_DIR / f'shaped_gps_optimised_{suffix}.{ext}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved: {path}')
        plt.close(fig)


if __name__ == '__main__':
    main()
