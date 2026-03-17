"""
QSP vs GPS vs equiangular on the three-level Lambda clock.

QSP protocol: n instantaneous pulses R(theta_j, phi_j) in the xy-plane,
separated by (n-1) equal free-evolution gaps tau_free ≈ T/(n-1).
Large fixed Omega_fast = 20*pi so pulse durations tau_j = theta_j/Omega_fast
are small relative to tau_free.

Noise models compared:
  1. White noise                   S(w) = 1
  2. 1/f noise                     S(w) = 1/|w|
  3. High-pass white (w_c = 2)     S(w) = theta(|w| - 2)

Protocols:
  Ramsey          (= QSP n=2 with theta=[pi/2, pi/2], phi=[0, 0])
  QSP  n=3, 5, 9  (optimised)
  GPS  m=1, 8     (single continuous drive, delta=0)
  Equiangular N=4 (white-noise optimised, omega_max=2)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
from quantum_pulse_suite.core.three_level_filter import (
    fft_three_level_filter, analytic_filter, detuning_sensitivity,
    default_omega_cutoff,
)
from quantum_pulse_suite.analysis.pulse_optimizer import (
    optimize_equiangular_sequence,
    optimize_qsp_sequence,
    build_qsp_3level,
    PulseOptimizationResult,
    QSPOptimizationResult,
    white_noise_psd,
    one_over_f_psd,
    high_pass_psd,
)

T          = 2 * np.pi
OMEGA_FAST = 20.0 * np.pi
OMEGA_CUTOFF = default_omega_cutoff(T)
N_FFT      = 4096   # time samples for FFT Kubo integrals
OMEGA_PLOT = np.logspace(-2, np.log10(30), 800)   # log-spaced for analytic plots
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'
CACHE_PREFIX = 'qsp_opt_cache'


# =============================================================================
# Optimization cache helpers
# =============================================================================

def _sanitize(label):
    """Turn a noise label into a safe key component."""
    return label.replace('/', 'f').replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')


def find_latest_cache_path():
    """Return the newest timestamped QSP cache, or None if absent."""
    matches = sorted(OUTPUT_DIR.glob(f'{CACHE_PREFIX}_*.npz'))
    if matches:
        return matches[-1]
    legacy = OUTPUT_DIR / 'qsp_opt_cache.npz'
    if legacy.exists():
        return legacy
    return None


def load_opt_cache(system, noise_labels, qsp_ns):
    """
    Load cached QSP results and rebuild sequences.

    Equiangular N=4 is loaded from the equiangular cache (run_equiangular_optimization.py).
    Returns (res_eq4, qsp_results) if both caches exist, else None.
    """
    from plot_protocol_comparison import find_latest_equiangular_cache

    # ── QSP cache ─────────────────────────────────────────────────────────────
    qsp_cache_path = find_latest_cache_path()
    if qsp_cache_path is None:
        return None
    try:
        c = np.load(str(qsp_cache_path))
    except Exception as e:
        print(f'QSP cache load failed ({e}).')
        return None

    qsp_results = {n: {} for n in qsp_ns}
    for n in qsp_ns:
        for nlabel in noise_labels:
            k = f'qsp_n{n}_{_sanitize(nlabel)}'
            thetas     = c[f'{k}_thetas']
            phis       = c[f'{k}_phis']
            omega_fast = float(c[f'{k}_omega_fast'])
            tau_free   = float(c[f'{k}_tau_free'])
            seq = build_qsp_3level(system, T, n, thetas, phis, omega_fast)
            qsp_results[n][nlabel] = QSPOptimizationResult(
                n=n, thetas=thetas, phis=phis,
                omega_fast=omega_fast, tau_free=tau_free,
                sensitivity_sq=float(c[f'{k}_sensitivity_sq']),
                noise_var=float(c[f'{k}_noise_var']),
                sigma_nu=float(c[f'{k}_sigma_nu']),
                objective_score=float(c[f'{k}_objective_score']) if f'{k}_objective_score' in c else float(c[f'{k}_sigma_nu']),
                objective_mode=c[f'{k}_objective_mode'].item() if f'{k}_objective_mode' in c else 'sigma_nu',
                noise_label=nlabel, seq=seq,
            )
    print(f'Loaded QSP results from cache: {qsp_cache_path}')

    # ── Equiangular N=4 from equiangular cache ────────────────────────────────
    eq_cache_path = find_latest_equiangular_cache()
    if eq_cache_path is None:
        raise FileNotFoundError(
            'No equiangular cache found. Run scripts/run_equiangular_optimization.py first.'
        )
    ec = np.load(str(eq_cache_path), allow_pickle=True)
    omega  = float(ec['eq_N4_white_omega'])
    phases = ec['eq_N4_white_phases']
    tau    = T / len(phases)
    from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
    seq_eq4 = MultiLevelPulseSequence(system, system.probe)
    for phi in phases:
        seq_eq4.add_continuous_pulse(omega, [np.cos(phi), np.sin(phi), 0.0], 0.0, tau)
    seq_eq4.compute_polynomials()
    res_eq4 = PulseOptimizationResult(
        omega=omega, phases=phases,
        sensitivity_sq=float(ec['eq_N4_white_sens_sq']),
        noise_var=float(ec['eq_N4_white_noise_var']),
        sigma_nu=float(ec['eq_N4_white_sigma_nu']),
        objective_score=float(ec['eq_N4_white_objective_score']) if 'eq_N4_white_objective_score' in ec else float(ec['eq_N4_white_sigma_nu']),
        objective_mode=ec['eq_N4_white_objective_mode'].item() if 'eq_N4_white_objective_mode' in ec else 'sigma_nu',
        noise_label='white', seq=seq_eq4,
    )
    print(f'Loaded equiangular N=4 from cache: {eq_cache_path}')

    return res_eq4, qsp_results


# =============================================================================
# Reference sequences
# =============================================================================

def build_gps(system, m):
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(2 * np.pi * m / T, [1, 0, 0], 0.0, T)
    seq.compute_polynomials()
    return seq


def build_ramsey(system):
    """QSP n=2 with two pi/2 pulses."""
    thetas = np.array([np.pi / 2, np.pi / 2])
    phis   = np.array([0.0, 0.0])
    return build_qsp_3level(system, T, 2, thetas, phis, OMEGA_FAST)


def get_filter(seq):
    """Return (sens_sq, freqs, Fe) via analytic Fourier integrals on log-spaced grid."""
    _, ss = detuning_sensitivity(seq)
    fr, Fe = analytic_filter(seq, OMEGA_PLOT, m_y=1.0)
    return ss, fr, Fe


def get_filter_fft(seq):
    """Return (sens_sq, freqs, Fe) on dense FFT grid for accurate Kubo integrals."""
    _, ss = detuning_sensitivity(seq)
    fr, Fe, _, _ = fft_three_level_filter(seq, n_samples=N_FFT, pad_factor=4, m_y=1.0)
    return ss, fr, Fe


def sigma_nu_under(ss, fr, Fe, S_func):
    """Return sigma_nu = noise_var / sens_sq for given noise PSD."""
    mask = fr >= OMEGA_CUTOFF
    noise_var = float(simpson(Fe[mask] * S_func(fr[mask]), x=fr[mask]) / (2 * np.pi))
    return noise_var / ss if (ss > 0 and noise_var > 0) else 0.0


def noise_var_under(fr, Fe, S_func):
    """Return band-limited noise variance for the supplied PSD."""
    mask = fr >= OMEGA_CUTOFF
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(simpson(Fe[mask] * S_func(fr[mask]), x=fr[mask]) / (2 * np.pi))


def objective_score_under(ss, fr, Fe, S_func, ss_ref, noise_ref, weight=1.0):
    """Return Ramsey-normalized difference score for a protocol under one PSD."""
    noise_var = noise_var_under(fr, Fe, S_func)
    if ss_ref <= 0.0 or noise_ref <= 0.0:
        return 0.0
    return ss / ss_ref - weight * noise_var / noise_ref


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    S_w  = white_noise_psd()
    S_f  = one_over_f_psd()
    S_hp = high_pass_psd(omega_c=2.0)   # white noise only above omega=2

    noise_labels = ['White', '1/f', 'High-pass (w_c=2)']
    S_funcs      = [S_w, S_f, S_hp]

    # ── Reference protocols ───────────────────────────────────────────────────
    print('Building reference sequences ...', flush=True)
    seq_R    = build_ramsey(system)
    seq_gps1 = build_gps(system, 1)
    seq_gps8 = build_gps(system, 8)

    qsp_ns     = [4, 8, 13]
    qsp_colors = {4: 'C4', 8: 'C5', 13: 'C6'}

    # ── Load from cache (required; run scripts/run_optimization.py to generate) ─
    cached = load_opt_cache(system, noise_labels, qsp_ns)
    if cached is None:
        raise FileNotFoundError(
            f'No optimization cache found matching {CACHE_PREFIX}_*.npz in {OUTPUT_DIR}.\n'
            f'Run:  python scripts/run_optimization.py  to generate it.'
        )
    res_eq4, qsp_results = cached
    print(f'  Equiangular: Omega*T={res_eq4.omega*T:.4f}  '
          f'phases={np.array2string(res_eq4.phases, precision=3)}  '
          f'opt_score={res_eq4.objective_score:.3f}')
    for n in qsp_ns:
        for nlabel in noise_labels:
            r = qsp_results[n][nlabel]
            print(f'  QSP n={n}, {nlabel}: tau_free={r.tau_free:.4f}  '
                  f'opt_score={r.objective_score:.3f}  sigma_nu={r.sigma_nu:.3f}')

    seq_eq4 = res_eq4.seq

    refs = [
        ('Ramsey',          seq_R,    'C3', '--'),
        ('GPS m=1',         seq_gps1, 'C0', '-.'),
        ('GPS m=8',         seq_gps8, 'C2', '-.'),
        ('Equiangular N=4', seq_eq4,  'C1', '-'),
    ]
    ref_data     = {lbl: get_filter(seq)     for lbl, seq, _, _ in refs}
    ref_data_fft = {lbl: get_filter_fft(seq) for lbl, seq, _, _ in refs}

    # ── Full sigma_nu table ───────────────────────────────────────────────────
    print('\n--- sigma_nu table ---')
    hdr = f'{"Protocol":<22}' + ''.join(f' {nl:>22}' for nl in noise_labels)
    print(hdr)
    print('-' * len(hdr))

    # Reference rows (use FFT grid for accurate Kubo integrals)
    for lbl, seq, _, _ in refs:
        ss, fr, Fe = ref_data_fft[lbl]
        foms = [sigma_nu_under(ss, fr, Fe, S) for S in S_funcs]
        print(f'{lbl:<22}' + ''.join(f' {f:>22.3f}' for f in foms))

    # QSP rows (use FFT grid for accurate Kubo integrals)
    for n in qsp_ns:
        label = f'QSP n={n}'
        row_foms = []
        for S_func, nlabel in zip(S_funcs, noise_labels):
            r = qsp_results[n][nlabel]
            ss, fr, Fe = get_filter_fft(r.seq)
            row_foms.append(sigma_nu_under(ss, fr, Fe, S_func))
        print(f'{label:<22}' + ''.join(f' {f:>22.3f}' for f in row_foms))

    print('\n--- post-hoc objective-score table ---')
    print(hdr)
    print('-' * len(hdr))
    ref_norms = {}
    ss_ref, fr_ref, Fe_ref = ref_data_fft['Ramsey']
    for S_func, nlabel in zip(S_funcs, noise_labels):
        ref_norms[nlabel] = (ss_ref, noise_var_under(fr_ref, Fe_ref, S_func))

    for lbl, seq, _, _ in refs:
        ss, fr, Fe = ref_data_fft[lbl]
        scores = [
            objective_score_under(ss, fr, Fe, S_func, *ref_norms[nlabel])
            for S_func, nlabel in zip(S_funcs, noise_labels)
        ]
        print(f'{lbl:<22}' + ''.join(f' {s:>22.3f}' for s in scores))

    for n in qsp_ns:
        label = f'QSP n={n}'
        row_scores = []
        for S_func, nlabel in zip(S_funcs, noise_labels):
            r = qsp_results[n][nlabel]
            ss, fr, Fe = get_filter_fft(r.seq)
            row_scores.append(objective_score_under(ss, fr, Fe, S_func, *ref_norms[nlabel]))
        print(f'{label:<22}' + ''.join(f' {s:>22.3f}' for s in row_scores))

    # ── Filter function plots ─────────────────────────────────────────────────
    # One figure per noise model: filter functions of all protocols at the
    # optimum for THAT noise model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, S_func, nlabel in zip(axes, S_funcs, noise_labels):
        # References
        for lbl, seq, color, ls in refs:
            ss, fr, Fe = ref_data[lbl]
            ax.loglog(fr, Fe / T**2 + 1e-20, color=color, lw=1.5, ls=ls, label=lbl)

        # QSP optimised for THIS noise
        for n in qsp_ns:
            r   = qsp_results[n][nlabel]
            ss, fr, Fe = get_filter(r.seq)
            ax.loglog(fr, Fe / T**2 + 1e-20,
                      color=qsp_colors[n], lw=2.0,
                      label=f'QSP n={n}')

        # Noise PSD (scaled for visibility)
        w_vis = np.linspace(3e-2, 30, 2000)
        s_vis = S_func(w_vis)
        s_vis = s_vis / np.max(s_vis + 1e-30) * 0.3  # normalise to 30% of y-range
        ax.loglog(w_vis, s_vis + 1e-20, 'k:', lw=1, alpha=0.4, label=r'$S(\omega)$ (scaled)')

        ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=10)
        ax.set_ylabel(r'$F(\omega)/T^2$', fontsize=10)
        ax.set_title(f'Noise: {nlabel}', fontsize=11)
        ax.set_xlim([3e-2, 30])
        ax.set_ylim([1e-8, 1.5])
        ax.grid(True, alpha=0.3, which='both')
        if ax is axes[0]:
            ax.legend(fontsize=7, loc='lower left')

    fig.suptitle(
        r'Filter functions at each optimum: QSP vs GPS vs equiangular'
        '\n'
        r'Three-level Lambda clock, $T = 2\pi$, QSP $\Omega_{\rm fast} = 20\pi$',
        fontsize=12)

    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'qsp_comparison.{ext}',
                    dpi=300, bbox_inches='tight')
    print(f'\nSaved qsp_comparison')

    # ── FOM bar chart ─────────────────────────────────────────────────────────
    all_labels = (
        ['Ramsey', 'GPS m=1', 'GPS m=8', 'Eq N=4']
        + [f'QSP n={n}' for n in qsp_ns]
    )
    all_seqs_by_noise = {nlabel: [] for nlabel in noise_labels}
    all_scores_by_noise = {nlabel: [] for nlabel in noise_labels}

    for nlabel, S_func in zip(noise_labels, S_funcs):
        for lbl, seq, _, _ in refs:
            ss, fr, Fe = ref_data_fft[lbl]
            all_seqs_by_noise[nlabel].append(sigma_nu_under(ss, fr, Fe, S_func))
            all_scores_by_noise[nlabel].append(objective_score_under(ss, fr, Fe, S_func, *ref_norms[nlabel]))
        for n in qsp_ns:
            r = qsp_results[n][nlabel]
            ss, fr, Fe = get_filter_fft(r.seq)
            all_seqs_by_noise[nlabel].append(sigma_nu_under(ss, fr, Fe, S_func))
            all_scores_by_noise[nlabel].append(objective_score_under(ss, fr, Fe, S_func, *ref_norms[nlabel]))

    x   = np.arange(len(all_labels))
    w   = 0.25
    fig2, ax2 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    colors_bar = ['steelblue', 'seagreen', 'darkorange']
    for i, (nlabel, color) in enumerate(zip(noise_labels, colors_bar)):
        ax2.bar(x + (i - 1) * w, all_seqs_by_noise[nlabel], w,
                label=nlabel, color=color, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel(r'$\sigma^2_\nu$', fontsize=11)
    ax2.set_title(
        r'$\sigma^2_\nu$ comparison: QSP vs GPS vs equiangular ($T = 2\pi$)'
        '\n'
        r'Each QSP sequence optimised independently for each noise model',
        fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

    for ext in ['pdf', 'png']:
        fig2.savefig(OUTPUT_DIR / f'qsp_fom_bars.{ext}',
                     dpi=300, bbox_inches='tight')
    print('Saved qsp_fom_bars')

    fig3, ax3 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for i, (nlabel, color) in enumerate(zip(noise_labels, colors_bar)):
        ax3.bar(x + (i - 1) * w, all_scores_by_noise[nlabel], w,
                label=nlabel, color=color, alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels(all_labels, rotation=15, ha='right', fontsize=9)
    ax3.set_ylabel('Objective score', fontsize=11)
    ax3.set_title(
        'Post-hoc objective-score comparison: QSP vs GPS vs equiangular'
        '\n'
        'Score = sens/sens_Ramsey - noise/noise_Ramsey for each noise model',
        fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)

    for ext in ['pdf', 'png']:
        fig3.savefig(OUTPUT_DIR / f'qsp_objective_bars.{ext}',
                     dpi=300, bbox_inches='tight')
    print('Saved qsp_objective_bars')

    plt.close('all')


if __name__ == '__main__':
    main()
