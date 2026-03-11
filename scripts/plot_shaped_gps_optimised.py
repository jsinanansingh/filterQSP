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
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson
from scipy.optimize import differential_evolution, minimize

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
from quantum_pulse_suite.core.three_level_filter import (
    fft_three_level_filter, detuning_sensitivity, analytic_filter,
    default_omega_cutoff,
)
from quantum_pulse_suite.analysis.pulse_optimizer import white_noise_psd, one_over_f_psd

# =============================================================================
# Global config
# =============================================================================

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi
N_DISC     = 64          # segments for shaped sequence (optimisation)
N_DISC_FIN = 256         # segments for final evaluation
N_FFT      = 1024        # fft samples during optimisation
N_FFT_FIN  = 4096        # fft samples for final evaluation
PAD        = 4           # must match final evaluation to avoid bias
FREQS_PLOT = np.logspace(-1.5, np.log10(30), 600)
N_FREE     = 4           # free Fourier coefficients b_2..b_{N_FREE+1}
OMEGA_CUTOFF = default_omega_cutoff(T)

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'
CACHE_DIR  = OUTPUT_DIR / 'cache'   # timestamped runs stored here


def _latest_cache(required_key='m1_Opt0_sigma_nu_w'):
    """Return path to most recent cache file that contains required_key, or None."""
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
    """Save data to a timestamped cache file in CACHE_DIR."""
    from datetime import datetime
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = CACHE_DIR / f'shaped_gps_opt_cache_{ts}.npz'
    np.savez(str(path), **data)
    return path

S_white    = white_noise_psd()
S_1_over_f = one_over_f_psd()


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
    """Return an envelope fn from free coefficients b_2..b_{N+1}."""
    b1     = 1.0 - float(np.sum(b_free))
    b_all  = np.concatenate([[b1], np.asarray(b_free)])   # shape (N_FREE+1,)
    ks     = np.arange(1, len(b_all) + 1)                 # 1..N_FREE+1

    def fn(ts, T_total, omega_mean):
        phases = 2.0 * np.pi * np.outer(ks, ts) / T_total  # (N, n_pts)
        return omega_mean * (1.0 - b_all @ np.cos(phases))
    return fn


# =============================================================================
# Sequence builder (delta baked in)
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
# sigma_nu evaluator
# =============================================================================

def compute_sigma_nu(seq, n_fft=N_FFT, pad=PAD):
    _, sens_sq = detuning_sensitivity(seq)
    if sens_sq < 1e-20:
        return np.inf, 0.0, 0.0, None
    freqs, Fe, _, _ = fft_three_level_filter(seq, n_samples=n_fft,
                                              pad_factor=pad, m_y=1.0)
    mask = freqs >= OMEGA_CUTOFF
    if np.count_nonzero(mask) < 2:
        return np.inf, sens_sq, 0.0, (freqs, Fe)
    noise_w = float(simpson(Fe[mask] * S_white(freqs[mask]), x=freqs[mask]) / (2*np.pi))
    sigma_nu_w = noise_w / sens_sq if noise_w > 0 else np.inf
    return sigma_nu_w, sens_sq, noise_w, (freqs, Fe)


def negativity_penalty(envelope_fn, omega_mean, n_check=256):
    """Penalty proportional to integral of negative part of Omega(t)."""
    ts  = (np.arange(n_check) + 0.5) * T / n_check
    om  = envelope_fn(ts, T, omega_mean)
    neg = np.minimum(om, 0.0)
    return float(-np.sum(neg) * T / n_check)   # >= 0


# =============================================================================
# Step 1: optimise delta for a fixed envelope shape
# =============================================================================

def optimise_delta(system, omega_mean, envelope_fn, delta_max,
                   n_scan=60, label=''):
    """Scan then polish to find best delta for a fixed shape."""
    deltas = np.linspace(-delta_max, delta_max, n_scan)
    best_fom, best_d = np.inf, 0.0

    for d in deltas:
        pen = negativity_penalty(envelope_fn, omega_mean)
        if pen > 1e-3:      # skip obviously bad shapes during scan
            continue
        seq = build_seq(system, omega_mean, envelope_fn, d)
        sigma_nu, _, _, _ = compute_sigma_nu(seq)
        if sigma_nu < best_fom:
            best_fom, best_d = sigma_nu, d

    # Polish with scalar minimizer
    def sigma_nu_obj(d):
        seq = build_seq(system, omega_mean, envelope_fn, float(d))
        sigma_nu, _, _, _ = compute_sigma_nu(seq)
        return sigma_nu

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(sigma_nu_obj,
                          bounds=(best_d - delta_max/5, best_d + delta_max/5),
                          method='bounded',
                          options={'xatol': 1e-4})
    best_d   = float(res.x)
    best_fom = float(res.fun)
    if label:
        print(f'    {label}: delta*={best_d:.4f}  sigma_nu={best_fom:.4f}')
    return best_d, best_fom


# =============================================================================
# Step 2: optimise envelope (Fourier coefficients) + delta jointly
# =============================================================================

def optimise_envelope(system, omega_mean, delta_max, seed=42, label='',
                      popsize=12, maxiter=250, n_restarts=3):
    """
    Joint optimisation over free Fourier coefficients b_2..b_{N_FREE+1}
    and operating detuning delta.
    """
    b_bounds = [(-1.5, 1.5)] * N_FREE
    d_bound  = [(-delta_max, delta_max)]
    bounds   = b_bounds + d_bound

    def objective(x):
        b_free = x[:N_FREE]
        delta  = float(x[N_FREE])
        fn     = make_cosine_envelope(b_free)
        # Penalise negative envelope
        pen = negativity_penalty(fn, omega_mean) * 1e4
        seq = build_seq(system, omega_mean, fn, delta)
        sigma_nu, _, _, _ = compute_sigma_nu(seq)
        return sigma_nu + pen

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

    b_opt   = best_x[:N_FREE]
    d_opt   = float(best_x[N_FREE])
    fn_opt  = make_cosine_envelope(b_opt)
    sigma_nu_opt = best_val
    b1_opt  = 1.0 - float(np.sum(b_opt))
    if label:
        b_all = np.concatenate([[b1_opt], b_opt])
        print(f'    {label}: delta*={d_opt:.4f}  sigma_nu={sigma_nu_opt:.4f}')
        print(f'      b = [{", ".join(f"{v:.4f}" for v in b_all)}]')
    return d_opt, sigma_nu_opt, fn_opt, b_opt


# =============================================================================
# Step 2b: optimise envelope at delta=0 fixed
# =============================================================================

def optimise_envelope_delta0(system, omega_mean, seed=42, label='',
                              popsize=12, maxiter=250, n_restarts=3):
    """
    Optimise free Fourier coefficients b_2..b_{N_FREE+1} with delta fixed at 0.
    """
    bounds = [(-1.5, 1.5)] * N_FREE

    def objective(b_free):
        fn  = make_cosine_envelope(b_free)
        pen = negativity_penalty(fn, omega_mean) * 1e4
        seq = build_seq(system, omega_mean, fn, 0.0)
        sigma_nu, _, _, _ = compute_sigma_nu(seq)
        return sigma_nu + pen

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
        print(f'    {label}: sigma_nu={best_val:.4f}  (delta=0 fixed)')
        print(f'      b = [{", ".join(f"{v:.4f}" for v in b_all)}]')
    return fn_opt, b_opt


# =============================================================================
# Final high-resolution evaluation + filter function
# =============================================================================

def final_eval(system, omega_mean, envelope_fn, delta, label=''):
    seq = build_seq(system, omega_mean, envelope_fn, delta, n_disc=N_DISC_FIN)
    _, sens_sq = detuning_sensitivity(seq)
    freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
        seq, n_samples=N_FFT_FIN, pad_factor=4, m_y=1.0)
    mask = freqs_fft >= OMEGA_CUTOFF
    kubo_w = float(simpson(Fe_fft[mask] * S_white(freqs_fft[mask]),    x=freqs_fft[mask]) / (2*np.pi))
    kubo_f = float(simpson(Fe_fft[mask] * S_1_over_f(freqs_fft[mask]), x=freqs_fft[mask]) / (2*np.pi))
    sigma_nu_w = kubo_w / sens_sq if kubo_w > 0 else np.inf
    sigma_nu_f = kubo_f / sens_sq if kubo_f > 0 else np.inf
    if label:
        print(f'  {label:<42}  sens={sens_sq:.4f}  sigma_nu_w={sigma_nu_w:.4f}'
              f'  sigma_nu_1/f={sigma_nu_f:.4f}  delta={delta:.4f}')
    # Keep only frequencies up to plot range to avoid huge arrays
    mask = freqs_fft <= 35.0
    return dict(sens_sq=sens_sq, kubo_w=kubo_w, sigma_nu_w=sigma_nu_w,
                sigma_nu_f=sigma_nu_f, delta=delta,
                freqs=freqs_fft[mask], Fe=Fe_fft[mask])


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

    results   = {}   # results[m][label]
    opt_envs  = {}   # opt_envs[m]  = (fn_opt,  b_opt)   -- free delta
    opt0_envs = {}   # opt0_envs[m] = (fn_opt0, b_opt0)  -- delta=0 fixed

    # ── Load or compute ───────────────────────────────────────────────────────
    # Keys needed in cache
    _all_names = [s for s, _ in fixed_shapes] + ['Opt', 'Opt0']
    _cache_keys = ('sens_sq', 'kubo_w', 'sigma_nu_w', 'sigma_nu_f', 'delta', 'freqs', 'Fe')

    _cache_path = _latest_cache(required_key='m1_Opt0_sigma_nu_w')
    cache_ok    = _cache_path is not None

    if cache_ok:
        print(f'Loading cached results from {_cache_path}')
        cache = np.load(str(_cache_path), allow_pickle=True)
        ramsey = {k: cache[f'ramsey_{k}'] for k in
                  ('sens_sq', 'sigma_nu_w', 'freqs', 'Fe')}
        ramsey = {k: float(v) if v.ndim == 0 else v for k, v in ramsey.items()}
        for m_cyc, _, _ in gps_cases:
            results[m_cyc] = {}
            for name in _all_names:
                key = f'm{m_cyc}_{name}'
            results[m_cyc][name] = {
                k: float(cache[f'{key}_{k}']) if cache[f'{key}_{k}'].ndim == 0
                else cache[f'{key}_{k}']
                for k in _cache_keys
            }
            b_opt  = cache[f'm{m_cyc}_opt_b']
            b_opt0 = cache[f'm{m_cyc}_opt0_b']
            opt_envs[m_cyc]  = (make_cosine_envelope(b_opt),  b_opt)
            opt0_envs[m_cyc] = (make_cosine_envelope(b_opt0), b_opt0)
        print('  Done.')
    else:
        # ── Step 0: Ramsey baseline ───────────────────────────────────────────
        print('Ramsey baseline ...')
        seq_r = build_ramsey(system)
        _, sens_r = detuning_sensitivity(seq_r)
        freqs_r, Fe_r_fft, _, _ = fft_three_level_filter(
            seq_r, n_samples=N_FFT_FIN, pad_factor=4, m_y=1.0)
        mask_r = freqs_r >= OMEGA_CUTOFF
        kubo_r = float(simpson(Fe_r_fft[mask_r] * S_white(freqs_r[mask_r]), x=freqs_r[mask_r]) / (2*np.pi))
        mask_r = freqs_r <= 35.0
        ramsey = dict(sens_sq=sens_r, sigma_nu_w=kubo_r/sens_r,
                      freqs=freqs_r[mask_r], Fe=Fe_r_fft[mask_r])
        print(f'  Ramsey: sigma_nu_w={ramsey["sigma_nu_w"]:.4f}')

        for m_cyc, omega_mean, delta_max in gps_cases:
            results[m_cyc] = {}
            print(f'\n=== GPS m={m_cyc}  (Omega_mean={omega_mean:.4f}) ===')

            # ── Step 1: optimise delta for each fixed shape ───────────────────
            print('\n  Step 1 – optimise delta for fixed shapes:')
            for name, fn in fixed_shapes:
                d_opt, _ = optimise_delta(
                    system, omega_mean, fn, delta_max,
                    label=f'{name} m={m_cyc}')
                results[m_cyc][name] = final_eval(
                    system, omega_mean, fn, d_opt,
                    label=f'GPS m={m_cyc}  {name} (delta_opt)')

            # ── Step 2: optimise envelope + delta ─────────────────────────────
            print('\n  Step 2 – optimise envelope shape + delta:')
            t0 = time.time()
            d_opt, _, fn_opt, b_opt = optimise_envelope(
                system, omega_mean, delta_max,
                seed=17 + m_cyc, label=f'Opt-envelope m={m_cyc}',
                popsize=12, maxiter=250, n_restarts=4)
            print(f'    (optimisation took {time.time()-t0:.1f}s)')
            opt_envs[m_cyc] = (fn_opt, b_opt)
            results[m_cyc]['Opt'] = final_eval(
                system, omega_mean, fn_opt, d_opt,
                label=f'GPS m={m_cyc}  Opt-envelope (delta_opt)')

            # ── Step 2b: optimise envelope at delta=0 fixed ───────────────────
            print('\n  Step 2b – optimise envelope at delta=0:')
            t0 = time.time()
            fn_opt0, b_opt0 = optimise_envelope_delta0(
                system, omega_mean,
                seed=31 + m_cyc, label=f'Opt0-envelope m={m_cyc}',
                popsize=12, maxiter=250, n_restarts=4)
            print(f'    (optimisation took {time.time()-t0:.1f}s)')
            opt0_envs[m_cyc] = (fn_opt0, b_opt0)
            results[m_cyc]['Opt0'] = final_eval(
                system, omega_mean, fn_opt0, 0.0,
                label=f'GPS m={m_cyc}  Opt0-envelope (delta=0)')

        # ── Save cache ────────────────────────────────────────────────────────
        d = {}
        for k, v in ramsey.items():
            d[f'ramsey_{k}'] = np.array(v)
        for m_cyc, _, _ in gps_cases:
            for name in _all_names:
                key = f'm{m_cyc}_{name}'
                for k, v in results[m_cyc][name].items():
                    d[f'{key}_{k}'] = np.array(v)
            d[f'm{m_cyc}_opt_b']  = opt_envs[m_cyc][1]
            d[f'm{m_cyc}_opt0_b'] = opt0_envs[m_cyc][1]
        saved = _save_cache(d)
        print(f'\nCache saved to {saved}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n\n=== Summary ===')
    hdr = f'{"Protocol":<44} {"sigma_nu_w":>12} {"sigma_nu_1/f":>14} {"delta":>8}'
    print(hdr);  print('-'*len(hdr))
    print(f'  {"Ramsey":<42} {ramsey["sigma_nu_w"]:>12.4f}')
    for m_cyc, _, _ in gps_cases:
        for name in _all_names:
            r   = results[m_cyc][name]
            lbl = f'GPS m={m_cyc}  {name}'
            print(f'  {lbl:<42} {r["sigma_nu_w"]:>12.4f} {r["sigma_nu_f"]:>14.4f}'
                  f' {r["delta"]:>8.4f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    color_map = {'Square': 'C0', 'Hann': 'C9', 'Blackman': 'C1',
                 'Opt': 'C3', 'Opt0': 'C5'}
    ls_map    = {'Square': '-',  'Hann': '--', 'Blackman': ':',
                 'Opt': '-',  'Opt0': '-'}
    lw_map    = {'Square': 1.8,  'Hann': 2.0, 'Blackman': 2.0,
                 'Opt': 2.5,  'Opt0': 2.5}

    for ax, (m_cyc, omega_mean, _) in zip(axes, gps_cases):
        ax.loglog(ramsey['freqs'], ramsey['Fe'] / T**2 + 1e-20,
                  color='0.55', lw=1.5, label='Ramsey (ref)')

        for name in _all_names:
            r   = results[m_cyc][name]
            d_str = r'$\delta{=}0$' if name == 'Opt0' else rf'$\delta^*$={r["delta"]:.2f}'
            lbl = rf'{name}  {d_str}  $\sigma_{{\nu,w}}$={r["sigma_nu_w"]:.3f}'
            ax.loglog(r['freqs'], r['Fe'] / T**2 + 1e-20,
                      color=color_map[name], lw=lw_map[name],
                      ls=ls_map[name], label=lbl)

        # Inset: envelope shapes (Opt and Opt0 only — cleaner)
        ax_ins = ax.inset_axes([0.62, 0.55, 0.36, 0.38])
        ts = np.linspace(0, T, 300)
        ax_ins.plot(ts / np.pi, envelope_square(ts, T, omega_mean) / omega_mean,
                    color='C0', lw=1.2, ls='-', label='Square')
        fn_opt,  _ = opt_envs[m_cyc]
        fn_opt0, _ = opt0_envs[m_cyc]
        ax_ins.plot(ts / np.pi, fn_opt(ts, T, omega_mean)  / omega_mean,
                    color='C3', lw=2.0, ls='-', label='Opt')
        ax_ins.plot(ts / np.pi, fn_opt0(ts, T, omega_mean) / omega_mean,
                    color='C5', lw=2.0, ls='-', label=r'Opt$\delta{=}0$')
        ax_ins.set_xlabel(r'$t/\pi$', fontsize=7)
        ax_ins.set_ylabel(r'$\Omega/\bar\Omega$', fontsize=7)
        ax_ins.tick_params(labelsize=6)
        ax_ins.set_xlim(0, 2); ax_ins.set_ylim(bottom=0)
        ax_ins.legend(fontsize=5.5, loc='upper center')
        ax_ins.grid(True, alpha=0.3)

        # Rabi harmonics
        for k in range(1, 6):
            if omega_mean * k <= 30:
                ax.axvline(omega_mean * k, color='0.7', lw=0.5, ls=':', alpha=0.5)

        ax.set_xlim([FREQS_PLOT[0], 30])
        ax.set_ylim([1e-9, 1.5])
        ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=11)
        ax.set_title(rf'GPS $m{{=}}{m_cyc}$  ($\bar\Omega={omega_mean:.1f}$)',
                     fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=7.5, loc='lower left')

    axes[0].set_ylabel(r'$F_e(\omega)\,/\,T^2$', fontsize=11)
    fig.suptitle(
        r'Pulse-shaped GPS: filter functions at optimised $\delta$'
        '\n'
        r'Solid "Opt" = optimised Fourier envelope  '
        r'($N_{\rm free}=' + str(N_FREE) + r'$ cosine terms + $\delta$)',
        fontsize=10)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'shaped_gps_optimised.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')


if __name__ == '__main__':
    main()
