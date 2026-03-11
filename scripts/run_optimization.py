"""
Run QSP and equiangular sequence optimizations and save results to cache.

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
    optimize_equiangular_sequence,
    optimize_qsp_sequence,
    build_qsp_3level,
)

# Import cache helpers from plot_qsp_comparison
sys.path.insert(0, str(Path(__file__).parent))
from plot_qsp_comparison import (
    save_opt_cache,
    white_noise_psd, one_over_f_psd, high_pass_psd,
    T, OMEGA_FAST, OMEGA_CUTOFF,
)

def build_ramsey(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


def main():
    system = ThreeLevelClock()

    S_w  = white_noise_psd()
    S_f  = one_over_f_psd()
    S_hp = high_pass_psd(omega_c=2.0)

    noise_labels = ['White', '1/f', 'High-pass (w_c=2)']
    S_funcs      = [S_w, S_f, S_hp]
    qsp_ns       = [3, 5, 9]

    # ── Equiangular N=4 ───────────────────────────────────────────────────────
    print('Optimising equiangular N=4 (white noise) ...', flush=True)
    res_eq4 = optimize_equiangular_sequence(
        system, T, N=4, noise_psd='white',
        omega_cutoff=OMEGA_CUTOFF,
        n_restarts=12, seed=7, n_fft=1024, pad_factor=2,
        popsize=20, maxiter=500,
    )
    print(f'  Omega*T={res_eq4.omega*T:.4f}  '
          f'phases={np.array2string(res_eq4.phases, precision=4)}', flush=True)
    print(f'  sigma_nu_white = {res_eq4.sigma_nu:.4f}', flush=True)

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
                n_restarts=nres, seed=42 + n,
                n_fft=1024, pad_factor=2,
                popsize=pop, maxiter=mit,
            )
            qsp_results[n][nlabel] = r
            print(f'    tau_free={r.tau_free:.4f}  sens={r.sensitivity_sq:.4f}  '
                  f'sigma_nu={r.sigma_nu:.4f}', flush=True)
            print(f'    thetas={np.array2string(r.thetas, precision=4)}')
            print(f'    phis  ={np.array2string(r.phis,   precision=4)}', flush=True)

    cache_path = save_opt_cache(res_eq4, qsp_results)
    print(f'\nDone. Cache saved to: {cache_path}')
    print('You can now run plot_qsp_comparison.py to generate plots instantly.')


if __name__ == '__main__':
    main()
