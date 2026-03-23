"""
Best-of comparison: old (omega_max=50) vs new (omega_max=251) QSP caches.

For each (n, noise_type), evaluates BOTH sequences with the correct
integration bounds (omega_min=2pi/T, omega_max=4*omega_fast~251) and
takes the winner (lower sigma_nu).

Outputs:
  - Winner sigma_nu table (for main_pra.tex Table I)
  - Exact thetas/phis for each winning sequence (for supplemental table)
  - LaTeX rows for supplemental table

Usage:
    python scripts/best_of_comparison.py
"""

import sys
import numpy as np
from pathlib import Path
from scipy.integrate import simpson

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.three_level_filter import analytic_filter, detuning_sensitivity
from quantum_pulse_suite.analysis.pulse_optimizer import (
    white_noise_psd, one_over_f_psd, high_pass_psd, build_qsp_3level,
)

# =============================================================================
# Parameters — must match Table I of the paper
# =============================================================================
T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi
OMEGA_MIN  = 2 * np.pi / T          # Fourier-limit cutoff (= 1.0)
OMEGA_MAX  = 4 * OMEGA_FAST         # ≈ 251 rad/s — captures all QSP pulse tails
N_OMEGA    = 2000

CACHE_DIR  = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'

OLD_QSP_CACHE = CACHE_DIR / 'qsp_opt_cache_20260317_154311.npz'   # omega_max=50
NEW_QSP_CACHE = CACHE_DIR / 'qsp_opt_cache_20260320_170624.npz'   # omega_max=251

QSP_NS = [4, 8, 13]

# (noise_label, cache_key, S_func)
NOISE_SPECS = [
    ('White',            'White',              white_noise_psd()),
    ('1/f',              '1ff',                one_over_f_psd()),
    ('High-pass (wc=2)', 'High-pass_w_c2',     high_pass_psd(omega_c=2.0)),
]

omega_grid = np.logspace(np.log10(OMEGA_MIN), np.log10(OMEGA_MAX), N_OMEGA)
system     = ThreeLevelClock()


# =============================================================================
# Helpers
# =============================================================================

def sigma_nu_from_seq(seq, S_func):
    _, sens_sq = detuning_sensitivity(seq)
    if sens_sq < 1e-20:
        return np.inf, sens_sq
    _, Fe = analytic_filter(seq, omega_grid, m_y=1.0)
    noise_var = float(simpson(Fe * S_func(omega_grid), x=omega_grid) / (2 * np.pi))
    return noise_var / sens_sq, sens_sq


def load_qsp_entry(cache, n, cache_key):
    """Return (thetas, phis, omega_fast) or None if key missing."""
    k = f'qsp_n{n}_{cache_key}'
    if f'{k}_thetas' not in cache:
        return None
    return (np.asarray(cache[f'{k}_thetas'], dtype=float),
            np.asarray(cache[f'{k}_phis'],   dtype=float),
            float(cache[f'{k}_omega_fast']))


def fmt_arr(a, prec=4):
    return '[' + ',\\ '.join(f'{v:.{prec}f}' for v in a) + ']'


def fmt_arr_over_pi(a, prec=4):
    return '[' + ',\\ '.join(f'{v/np.pi:.{prec}f}' for v in a) + ']'


# =============================================================================
# Main
# =============================================================================

def main():
    old_cache = np.load(str(OLD_QSP_CACHE), allow_pickle=True)
    new_cache = np.load(str(NEW_QSP_CACHE), allow_pickle=True)
    print(f'Old cache: {OLD_QSP_CACHE.name}')
    print(f'New cache: {NEW_QSP_CACHE.name}')
    print()

    # Store winning results
    results = {}  # (n, noise_label) -> dict

    for n in QSP_NS:
        for noise_label, cache_key, S_func in NOISE_SPECS:
            old_entry = load_qsp_entry(old_cache, n, cache_key)
            new_entry = load_qsp_entry(new_cache, n, cache_key)

            rows = []
            for tag, entry in [('old', old_entry), ('new', new_entry)]:
                if entry is None:
                    continue
                thetas, phis, of = entry
                seq = build_qsp_3level(system, T, n, thetas, phis, of)
                snu, sens_sq = sigma_nu_from_seq(seq, S_func)
                tau_free = (T - np.sum(thetas) / of) / (n - 1)
                rows.append(dict(tag=tag, snu=snu, sens_sq=sens_sq,
                                 thetas=thetas, phis=phis, of=of,
                                 tau_free=tau_free))

            if not rows:
                continue

            winner = min(rows, key=lambda r: r['snu'])
            results[(n, noise_label)] = winner

    # ── Print comparison table ──────────────────────────────────────────────
    print('=' * 90)
    print('BEST-OF COMPARISON  (all evaluated with omega_min=2pi/T, omega_max=4*omega_fast)')
    print('=' * 90)
    print(f'{"n":>4}  {"Noise":<22}  {"Old σ²_ν":>12}  {"New σ²_ν":>12}  {"Winner":>6}')
    print('-' * 70)

    for n in QSP_NS:
        for noise_label, cache_key, S_func in NOISE_SPECS:
            old_entry = load_qsp_entry(old_cache, n, cache_key)
            new_entry = load_qsp_entry(new_cache, n, cache_key)

            old_snu = new_snu = np.nan
            if old_entry:
                seq = build_qsp_3level(system, T, n, *old_entry)
                old_snu, _ = sigma_nu_from_seq(seq, S_func)
            if new_entry:
                seq = build_qsp_3level(system, T, n, *new_entry)
                new_snu, _ = sigma_nu_from_seq(seq, S_func)

            if np.isnan(old_snu) or old_snu < new_snu:
                winner_tag = 'old' if not np.isnan(old_snu) else '---'
            else:
                winner_tag = 'new'

            print(f'  n={n}  {noise_label:<22}  {old_snu:12.4e}  {new_snu:12.4e}  {winner_tag:>6}')

    # ── Print winner sigma_nu table (for Table I) ───────────────────────────
    print()
    print('=' * 90)
    print('WINNER TABLE I VALUES  (use these in main_pra.tex)')
    print('=' * 90)
    noise_labels = [nl for nl, _, _ in NOISE_SPECS]
    print(f'{"n":>4}  {"Fe(0) [white]":>14}' + ''.join(f'{nl:>20}' for nl in noise_labels))
    print('-' * 80)
    for n in QSP_NS:
        fe0 = results.get((n, 'White'), {}).get('sens_sq', np.nan)
        line = f'  n={n}  {fe0:>14.4f}'
        for noise_label in noise_labels:
            r = results.get((n, noise_label), {})
            line += f'{r.get("snu", np.nan):>20.4e}'
        print(line)

    # ── Print winning parameters ─────────────────────────────────────────────
    print()
    print('=' * 90)
    print('WINNING SEQUENCE PARAMETERS')
    print('=' * 90)

    for n in QSP_NS:
        print(f'\n--- QSP n={n} ---')
        for noise_label, _, _ in NOISE_SPECS:
            r = results.get((n, noise_label))
            if r is None:
                continue
            print(f'  [{noise_label}]  tag={r["tag"]}  sigma_nu={r["snu"]:.4e}'
                  f'  Fe(0)={r["sens_sq"]:.4f}  tau_free={r["tau_free"]:.4f}')
            print(f'    thetas/pi = {fmt_arr_over_pi(r["thetas"])}')
            print(f'    phis      = {fmt_arr(r["phis"])}')

    # ── LaTeX supplemental table ─────────────────────────────────────────────
    print()
    print('=' * 90)
    print('SUPPLEMENTAL TABLE  (LaTeX rows for tab:qsp_params_supp)')
    print('=' * 90)
    print('% White-noise-optimized sequence for each n')
    print()
    for n in QSP_NS:
        r = results.get((n, 'White'))
        if r is None:
            continue
        snu_white = r['snu']
        tau_free  = r['tau_free']
        thetas    = r['thetas']
        phis      = r['phis']
        # Format thetas/pi to 3 decimal places, phis to 3 decimal places
        th_str = ', '.join(f'{v/np.pi:.3f}' for v in thetas)
        ph_str = ', '.join(f'{v:.3f}' for v in phis)
        print(f'% n={n}  tau_free={tau_free:.4f}  sigma_nu_white={snu_white:.2e}')
        print(f'$\\theta/\\pi$: [{th_str}]')
        print(f'$\\phi$ (rad): [{ph_str}]')
        print()


if __name__ == '__main__':
    main()
