"""
Run QSP optimizations and save results to cache alongside a cached equiangular reference.

Run this script once (it takes several minutes).  After it completes,
plot_qsp_comparison.py will load from the cache and generate plots instantly.

Usage:
    python scripts/run_optimization.py
"""

import sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
from quantum_pulse_suite.analysis.pulse_optimizer import (
    optimize_qsp_sequence,
    PulseOptimizationResult,
)

# Import cache helpers from plot_qsp_comparison
sys.path.insert(0, str(Path(__file__).parent))
from plot_qsp_comparison import (
    save_opt_cache,
    white_noise_psd, one_over_f_psd, high_pass_psd,
    T, OMEGA_FAST, OMEGA_CUTOFF,
)
from plot_protocol_comparison import find_latest_equiangular_cache

OBJECTIVE_MODE = 'normalized_difference'
OBJECTIVE_WEIGHT = 1.0

def load_cached_eq4(system):
    """Load the white-noise N=4 equiangular result from the newest cache."""
    cache_path = find_latest_equiangular_cache()
    if cache_path is None:
        raise FileNotFoundError(
            'No equiangular cache found. Run scripts/run_equiangular_optimization.py first.'
        )
    c = np.load(str(cache_path), allow_pickle=True)
    omega = float(c['eq_N4_white_omega'])
    phases = np.asarray(c['eq_N4_white_phases'], dtype=float)
    tau = T / len(phases)
    seq = MultiLevelPulseSequence(system, system.probe)
    for phi in phases:
        seq.add_continuous_pulse(omega, [np.cos(phi), np.sin(phi), 0.0], 0.0, tau)
    seq.compute_polynomials()
    return PulseOptimizationResult(
        omega=omega,
        phases=phases,
        sensitivity_sq=float(c['eq_N4_white_sens_sq']),
        noise_var=float(c['eq_N4_white_noise_var']),
        sigma_nu=float(c['eq_N4_white_sigma_nu']),
        noise_label='white',
        seq=seq,
        objective_score=float(c['eq_N4_white_objective_score']) if 'eq_N4_white_objective_score' in c else float(c['eq_N4_white_sigma_nu']),
        objective_mode=c['eq_N4_white_objective_mode'].item() if 'eq_N4_white_objective_mode' in c else 'sigma_nu',
    ), cache_path


def main():
    system = ThreeLevelClock()

    S_w  = white_noise_psd()
    S_f  = one_over_f_psd()
    S_hp = high_pass_psd(omega_c=2.0)

    noise_labels = ['White', '1/f', 'High-pass (w_c=2)']
    S_funcs      = [S_w, S_f, S_hp]
    qsp_ns       = [3, 5, 9]

    # ── Equiangular N=4 from cache ────────────────────────────────────────────
    res_eq4, eq_cache_path = load_cached_eq4(system)
    print(f'Loaded equiangular N=4 from cache: {eq_cache_path}', flush=True)
    print(f'  Omega*T={res_eq4.omega*T:.4f}  '
          f'phases={np.array2string(res_eq4.phases, precision=4)}', flush=True)
    print(f'  score={res_eq4.objective_score:.4f}  sigma_nu_white={res_eq4.sigma_nu:.4f}', flush=True)

    # ── QSP sequences for each (n, noise) ────────────────────────────────────
    _qsp_budgets = {3: (10, 150, 4), 5: (10, 120, 3), 9: (8, 80, 2)}
    qsp_results  = {n: {} for n in qsp_ns}

    print('\nOptimising QSP sequences ...', flush=True)
    for n in qsp_ns:
        pop, mit, nres = _qsp_budgets[n]
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
            )
            qsp_results[n][nlabel] = r
            print(f'    tau_free={r.tau_free:.4f}  sens={r.sensitivity_sq:.4f}  '
                  f'score={r.objective_score:.4f}  sigma_nu={r.sigma_nu:.4f}', flush=True)
            print(f'    thetas={np.array2string(r.thetas, precision=4)}')
            print(f'    phis  ={np.array2string(r.phis,   precision=4)}', flush=True)

    cache_path = save_opt_cache(res_eq4, qsp_results)
    print(f'\nDone. Cache saved to: {cache_path}')
    print('You can now run plot_qsp_comparison.py to generate plots instantly.')


if __name__ == '__main__':
    main()
