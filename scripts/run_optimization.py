"""
Run QSP optimizations and save results to a timestamped cache.

Usage:
    python scripts/run_optimization.py
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.pulse_optimizer import (
    optimize_qsp_sequence,
    white_noise_psd, one_over_f_psd, high_pass_psd,
)

T            = 2 * np.pi
OMEGA_FAST   = 20.0 * np.pi
OMEGA_CUTOFF = None  # FL-cutoff: 2*pi/T (Fourier limit — matches paper Table I)
CACHE_PREFIX = 'qsp_opt_cache'
OUTPUT_DIR   = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'

OBJECTIVE_MODE      = 'ramsey_normalized'
OBJECTIVE_WEIGHT    = 1.0     # unused for ramsey_normalized
OMEGA_MAX_ANALYTIC  = 4.0 * OMEGA_FAST   # ≈ 251 rad/s — captures pulse-peak tails
N_OMEGA             = 1024    # frequency quadrature points over [omega_min, OMEGA_MAX_ANALYTIC]

NOISE_SPECS = [
    ('White',             white_noise_psd()),
    ('1/f',               one_over_f_psd()),
    ('High-pass (w_c=1)', high_pass_psd(omega_c=1.0)),
    ('High-pass (w_c=2)', high_pass_psd(omega_c=2.0)),
]

QSP_NS      = [4, 8, 13, 16]
QSP_BUDGETS = {4: (10, 150, 4), 8: (10, 100, 3), 13: (8, 80, 2), 16: (10, 150, 4)}


def _sanitize(label):
    return label.replace('/', 'f').replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')


def save_qsp_cache(qsp_results):
    """Save QSP optimization results to a timestamped npz cache."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d = {'objective_mode': np.array(OBJECTIVE_MODE)}
    for n, noise_dict in qsp_results.items():
        for nlabel, r in noise_dict.items():
            k = f'qsp_n{n}_{_sanitize(nlabel)}'
            d[f'{k}_thetas']          = r.thetas
            d[f'{k}_phis']            = r.phis
            d[f'{k}_omega_fast']      = np.array(r.omega_fast)
            d[f'{k}_tau_free']        = np.array(r.tau_free)
            d[f'{k}_sensitivity_sq']  = np.array(r.sensitivity_sq)
            d[f'{k}_noise_var']       = np.array(r.noise_var)
            d[f'{k}_sigma_nu']        = np.array(r.sigma_nu)
            d[f'{k}_objective_score'] = np.array(r.objective_score)
            d[f'{k}_objective_mode']  = np.array(r.objective_mode)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = OUTPUT_DIR / f'{CACHE_PREFIX}_{ts}.npz'
    np.savez(str(path), **d)
    return path


def main():
    system       = ThreeLevelClock()
    noise_labels = [lbl for lbl, _ in NOISE_SPECS]
    S_funcs      = [S   for _,   S in NOISE_SPECS]
    qsp_results  = {n: {} for n in QSP_NS}

    print(f'Objective mode: {OBJECTIVE_MODE}', flush=True)
    print(f'QSP n values:   {QSP_NS}', flush=True)
    print('\nOptimising QSP sequences ...', flush=True)

    for n in QSP_NS:
        pop, mit, nres = QSP_BUDGETS[n]
        for S_func, nlabel in zip(S_funcs, noise_labels):
            print(f'  QSP n={n}, {nlabel} ...', flush=True)
            r = optimize_qsp_sequence(
                system, T, n,
                noise_psd=S_func,
                omega_fast=OMEGA_FAST,
                omega_cutoff=OMEGA_CUTOFF,
                objective_mode=OBJECTIVE_MODE,
                objective_weight=OBJECTIVE_WEIGHT,
                n_restarts=nres, seed=42 + n,
                n_fft=1024, pad_factor=2,
                popsize=pop, maxiter=mit,
                omega_max_analytic=OMEGA_MAX_ANALYTIC,
                n_omega=N_OMEGA,
            )
            qsp_results[n][nlabel] = r
            print(f'    tau_free={r.tau_free:.4f}  sens={r.sensitivity_sq:.4f}  '
                  f'score={r.objective_score:.4f}  sigma_nu={r.sigma_nu:.4f}', flush=True)
            print(f'    thetas={np.array2string(r.thetas, precision=4)}')
            print(f'    phis  ={np.array2string(r.phis,   precision=4)}', flush=True)

    cache_path = save_qsp_cache(qsp_results)
    print(f'\nDone. Cache saved to: {cache_path}')


if __name__ == '__main__':
    main()
