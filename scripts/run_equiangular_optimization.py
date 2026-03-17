"""
Run equiangular pulse optimizations for N=4, 8, 16 under three noise models.

Noise models (matching main_pra.tex Sec. II):
  1. White noise:    S(w) = 1
  2. 1/f noise:      S(w) = 1/|w|
  3. High-pass:      S(w) = theta(|w| - 2)  (w_c = 2 rad/s)

Saves results to figures/qubit_performance_plots/equiangular_opt_cache.npz
and prints a LaTeX table ready for pasting.

Usage:
    python scripts/run_equiangular_optimization.py
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.pulse_optimizer import (
    optimize_equiangular_sequence,
    white_noise_psd,
    one_over_f_psd,
    high_pass_psd,
)

T          = 2 * np.pi
OMEGA_CUTOFF = 2 * np.pi / T
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'
CACHE_PREFIX = 'equiangular_opt_cache'
OBJECTIVE_MODE = 'ramsey_normalized'
OBJECTIVE_WEIGHT = 1.0  # unused for inv_sens_plus_noise

# ── Noise models ──────────────────────────────────────────────────────────────
NOISE_SPECS = [
    ('white',       'White',              white_noise_psd()),
    ('1f',          '$1/f$',              one_over_f_psd()),
    ('highpass2',   'High-pass ($\\omega_c{=}2$)', high_pass_psd(omega_c=2.0)),
]

# ── Optimization budgets (larger N → larger parameter space) ──────────────────
# use_analytic=True (default) uses the fast analytic filter path.
# n_fft / pad_factor are ignored when use_analytic=True.
OPT_BUDGETS = {
    4:  dict(popsize=15, maxiter=400, n_restarts=8, seed=7),
    8:  dict(popsize=15, maxiter=400, n_restarts=5, seed=13),
    16: dict(popsize=15, maxiter=400, n_restarts=3, seed=17),
}


def _timestamp():
    """Return a filesystem-safe local timestamp."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def cache_path_for_timestamp():
    """Return a timestamped equiangular cache path."""
    return OUTPUT_DIR / f'{CACHE_PREFIX}_{_timestamp()}.npz'


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    N_values     = [4, 8, 16]
    noise_keys   = [k for k, _, _ in NOISE_SPECS]
    noise_labels = [lbl for _, lbl, _ in NOISE_SPECS]
    noise_funcs  = [S for _, _, S in NOISE_SPECS]

    # results[N][noise_key] = PulseOptimizationResult
    results = {N: {} for N in N_values}

    for N in N_values:
        bud = OPT_BUDGETS[N]
        for key, label, S_func in NOISE_SPECS:
            print(f'\nOptimising N={N}, {label} ...', flush=True)
            r = optimize_equiangular_sequence(
                system, T, N,
                noise_psd=S_func,
                omega_cutoff=OMEGA_CUTOFF,
                objective_mode=OBJECTIVE_MODE,
                objective_weight=OBJECTIVE_WEIGHT,
                **bud,
            )
            results[N][key] = r
            print(f'  Omega*T = {r.omega * T:.5f}', flush=True)
            print(f'  phases  = {np.array2string(r.phases, precision=4, separator=", ")}', flush=True)
            print(f'  sens_sq = {r.sensitivity_sq:.4f}  noise_var = {r.noise_var:.4e}  score = {r.objective_score:.3f}  sigma_nu = {r.sigma_nu:.3f}', flush=True)

    # ── Save cache ────────────────────────────────────────────────────────────
    d = {}
    for N in N_values:
        for key in noise_keys:
            r  = results[N][key]
            pk = f'eq_N{N}_{key}'
            d[f'{pk}_omega']       = np.array(r.omega)
            d[f'{pk}_phases']      = r.phases
            d[f'{pk}_sens_sq']     = np.array(r.sensitivity_sq)
            d[f'{pk}_noise_var']   = np.array(r.noise_var)
            d[f'{pk}_sigma_nu']    = np.array(r.sigma_nu)
            d[f'{pk}_objective_score'] = np.array(r.objective_score)
            d[f'{pk}_objective_mode']  = np.array(r.objective_mode)
    cache_path = cache_path_for_timestamp()
    np.savez(str(cache_path), **d)
    print(f'\nCache saved to {cache_path}', flush=True)

    # ── Print summary table ───────────────────────────────────────────────────
    print('\n\n--- Summary table ---')
    hdr = f'{"Protocol":<24}' + ''.join(f' {"sigma_nu " + lbl:>20}' for lbl in noise_labels)
    print(hdr)
    print('-' * len(hdr))
    for N in N_values:
        row = f'Equiangular N={N:<2}'
        for key in noise_keys:
            row += f' {results[N][key].sigma_nu:>20.3f}'
        print(row)

    print('\n\n--- Objective-score table ---')
    hdr = f'{"Protocol":<24}' + ''.join(f' {"score " + lbl:>20}' for lbl in noise_labels)
    print(hdr)
    print('-' * len(hdr))
    for N in N_values:
        row = f'Equiangular N={N:<2}'
        for key in noise_keys:
            row += f' {results[N][key].objective_score:>20.3f}'
        print(row)

    # ── Print LaTeX table fragment ────────────────────────────────────────────
    print('\n\n--- LaTeX rows ---')
    for N in N_values:
        r_w  = results[N]['white']
        r_f  = results[N]['1f']
        r_hp = results[N]['highpass2']
        omT  = r_w.omega * T   # use white-noise Omega for display
        phs  = r_w.phases
        pstr = '[' + ', '.join(f'{p:.3f}' for p in phs) + ']'
        print(f'Equiangular $N{{=}}{N}$ & {r_w.sigma_nu:.4f} & {r_f.sigma_nu:.4f} & {r_hp.sigma_nu:.4f} \\\\')
        print(f'  % OmegaT={omT:.4f}  phases={pstr}')

    # ── Print detailed parameters per noise case ──────────────────────────────
    print('\n\n--- Detailed parameters ---')
    for N in N_values:
        print(f'\nN = {N}')
        for key, label, _ in NOISE_SPECS:
            r = results[N][key]
            phs_str = np.array2string(r.phases, precision=4, separator=', ')
            print(f'  {label}:  OmegaT={r.omega*T:.5f}  score={r.objective_score:.3f}  sigma_nu={r.sigma_nu:.3f}')
            print(f'    phases = {phs_str}')

    return results


if __name__ == '__main__':
    main()
