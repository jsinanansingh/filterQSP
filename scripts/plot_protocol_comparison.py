"""
Filter function comparison: Ramsey, GPS m=1, equiangular N=4/8/16, QSP n=4/8/13.

One panel per noise spectrum (white, 1/f, high-pass).  For equiangular and
QSP, each panel shows the sequence optimised for THAT noise model.

Equiangular sequences are loaded from the newest equiangular_opt_cache_*.npz.
QSP sequences are loaded from the newest qsp_opt_cache_*.npz.

All protocols share total interrogation time T = 2*pi, delta=0.
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
    fft_three_level_filter,
    analytic_filter,
    detuning_sensitivity,
)
from quantum_pulse_suite.analysis.pulse_optimizer import (
    white_noise_psd,
    one_over_f_psd,
    high_pass_psd,
    build_qsp_3level,
)

# =============================================================================
# Parameters
# =============================================================================

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi
N_FFT      = 4096
OMEGA_PLOT = np.logspace(-2, np.log10(30), 800)
OMEGA_MIN  = 2.0 * np.pi / T

OMEGA_GPS1 = 2 * np.pi * 1 / T   # = 1.0

OUTPUT_DIR   = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'
CACHE_DIR    = OUTPUT_DIR
EQ_PREFIX    = 'equiangular_opt_cache'
QSP_PREFIX   = 'qsp_opt_cache'

# Noise panels: (eq_cache_key, qsp_noise_label, panel_title, S_func)
# eq_cache_key  matches keys in equiangular cache: 'white', '1f', 'highpass2'
# qsp_noise_label is what run_optimization.py used as nlabel (pre-sanitize)
NOISE_PANELS = [
    ('white',     'White',             'White noise',              white_noise_psd()),
    ('1f',        '1/f',              r'$1/f$ noise',              one_over_f_psd()),
    ('highpass2', 'High-pass (w_c=2)', r'High-pass ($\omega_c=2$)', high_pass_psd(omega_c=2.0)),
]

EQ_NS   = [4, 8, 16]
QSP_NS  = [4, 8, 13]

# Color scheme
COLOR_RAMSEY = '0.45'
COLOR_GPS1   = 'C0'
EQ_COLORS    = ['#E07020', '#C04000', '#802000']   # orange → dark brown
QSP_COLORS   = ['#2090D0', '#1060A0', '#083070']   # light → dark blue


# =============================================================================
# Cache helpers
# =============================================================================

def find_latest_equiangular_cache():
    """Return the newest timestamped equiangular cache, or a legacy path."""
    matches = sorted(CACHE_DIR.glob(f'{EQ_PREFIX}_*.npz'))
    if matches:
        return matches[-1]
    legacy = CACHE_DIR / f'{EQ_PREFIX}.npz'
    if legacy.exists():
        return legacy
    return None


def find_latest_qsp_cache():
    """Return the newest timestamped QSP cache, or None."""
    matches = sorted(CACHE_DIR.glob(f'{QSP_PREFIX}_*.npz'))
    return matches[-1] if matches else None


def _sanitize(label):
    return label.replace('/', 'f').replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')


# =============================================================================
# Sequence builders
# =============================================================================

def build_ramsey(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


def build_gps(system, omega):
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, T)
    seq.compute_polynomials()
    return seq


def build_equiangular(system, omega, phases):
    N   = len(phases)
    tau = T / N
    seq = MultiLevelPulseSequence(system, system.probe)
    for phi in phases:
        seq.add_continuous_pulse(omega, [np.cos(phi), np.sin(phi), 0.0], 0.0, tau)
    seq.compute_polynomials()
    return seq


# =============================================================================
# Evaluation
# =============================================================================

def get_filter(seq):
    """Return (sens_sq, freqs_plot, Fe_plot, freqs_fft, Fe_fft)."""
    _, sens_sq = detuning_sensitivity(seq)
    freqs_plot, Fe_plot = analytic_filter(seq, OMEGA_PLOT, m_y=1.0)
    freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
        seq, n_samples=N_FFT, pad_factor=4, m_y=1.0)
    return sens_sq, freqs_plot, Fe_plot, freqs_fft, Fe_fft


def noise_var(freqs_fft, Fe_fft, S_func):
    mask = freqs_fft >= OMEGA_MIN
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(simpson(Fe_fft[mask] * S_func(freqs_fft[mask]),
                         x=freqs_fft[mask]) / (2 * np.pi))


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    # ── Load caches ───────────────────────────────────────────────────────────
    eq_cache_path = find_latest_equiangular_cache()
    if eq_cache_path is None:
        raise FileNotFoundError(
            f'No equiangular cache found. Run scripts/run_equiangular_optimization.py first.'
        )
    eq_cache = np.load(str(eq_cache_path), allow_pickle=True)
    print(f'Equiangular cache: {eq_cache_path}')

    qsp_cache_path = find_latest_qsp_cache()
    qsp_cache = None
    if qsp_cache_path is not None:
        qsp_cache = np.load(str(qsp_cache_path), allow_pickle=True)
        print(f'QSP cache:         {qsp_cache_path}')
    else:
        print('WARNING: No QSP cache found — QSP curves will be omitted.')

    # ── Build noise-independent sequences ─────────────────────────────────────
    print('\nBuilding Ramsey and GPS m=1 ...')
    seq_ramsey = build_ramsey(system)
    seq_gps1   = build_gps(system, OMEGA_GPS1)

    res_ramsey = get_filter(seq_ramsey)
    res_gps1   = get_filter(seq_gps1)

    # ── Build per-noise sequences and evaluate ────────────────────────────────
    # panel_data[panel_idx] = {label: (sens_sq, freqs_plot, Fe_plot, freqs_fft, Fe_fft)}
    panel_seqs = []   # list of dicts per panel

    for eq_key, qsp_nlabel, panel_title, S_func in NOISE_PANELS:
        seqs = {}

        # Equiangular: load noise-specific result
        for N in EQ_NS:
            pk = f'eq_N{N}_{eq_key}'
            omega  = float(eq_cache[f'{pk}_omega'])
            phases = np.asarray(eq_cache[f'{pk}_phases'], dtype=float)
            seq    = build_equiangular(system, omega, phases)
            seqs[f'Eq N={N}'] = get_filter(seq)

        # QSP: load noise-specific result
        if qsp_cache is not None:
            for n in QSP_NS:
                k = f'qsp_n{n}_{_sanitize(qsp_nlabel)}'
                if f'{k}_thetas' not in qsp_cache:
                    continue
                thetas     = qsp_cache[f'{k}_thetas']
                phis       = qsp_cache[f'{k}_phis']
                omega_fast = float(qsp_cache[f'{k}_omega_fast'])
                seq        = build_qsp_3level(system, T, n, thetas, phis, omega_fast)
                seqs[f'QSP n={n}'] = get_filter(seq)

        panel_seqs.append(seqs)

    # ── Print summary table ───────────────────────────────────────────────────
    noise_labels = [title for _, _, title, _ in NOISE_PANELS]
    S_funcs      = [S     for _, _, _, S    in NOISE_PANELS]
    print()
    col = 14
    hdr = f'{"Protocol":<22}' + ''.join(f'{"sens_sq":>{col}}') + \
          ''.join(f'{f"snu_{eq_key}":>{col}}' for eq_key, *_ in NOISE_PANELS)
    print(hdr);  print('-' * len(hdr))

    def print_row(label, sens_sq, fft_data):
        freqs_fft, Fe_fft = fft_data
        row = f'{label:<22}{sens_sq:>{col}.4f}'
        for _, _, _, S_func in NOISE_PANELS:
            nv  = noise_var(freqs_fft, Fe_fft, S_func)
            snu = nv / sens_sq if (sens_sq > 0 and nv > 0) else 0.0
            row += f'{snu:>{col}.4f}'
        print(row)

    print_row('Ramsey',   res_ramsey[0], (res_ramsey[3], res_ramsey[4]))
    print_row('GPS m=1',  res_gps1[0],   (res_gps1[3],   res_gps1[4]))
    for i, (eq_key, *_) in enumerate(NOISE_PANELS):
        for N in EQ_NS:
            r = panel_seqs[i][f'Eq N={N}']
            print_row(f'Eq N={N} [{eq_key}]', r[0], (r[3], r[4]))
        for n in QSP_NS:
            if f'QSP n={n}' in panel_seqs[i]:
                r = panel_seqs[i][f'QSP n={n}']
                print_row(f'QSP n={n} [{eq_key}]', r[0], (r[3], r[4]))
        break  # one pass is enough for the table; noise-specific results printed below

    # ── Figure: 1 row × 3 panels ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True,
                             sharey=True)

    for ax, (eq_key, qsp_nlabel, panel_title, S_func), seqs in zip(axes, NOISE_PANELS, panel_seqs):

        # Ramsey and GPS (same in every panel)
        ax.loglog(res_ramsey[1], res_ramsey[2] / T**2 + 1e-20,
                  color=COLOR_RAMSEY, lw=2.0, ls='--', label='Ramsey', zorder=3)
        ax.loglog(res_gps1[1],   res_gps1[2]   / T**2 + 1e-20,
                  color=COLOR_GPS1,   lw=2.0, ls='-.', label=r'GPS $m{=}1$', zorder=3)

        # Equiangular (noise-specific)
        for N, color in zip(EQ_NS, EQ_COLORS):
            r = seqs[f'Eq N={N}']
            ax.loglog(r[1], r[2] / T**2 + 1e-20,
                      color=color, lw=2.2, ls='-',
                      label=rf'Eq $N{{\!=\!}}{N}$', zorder=4)

        # QSP (noise-specific)
        for n, color in zip(QSP_NS, QSP_COLORS):
            if f'QSP n={n}' not in seqs:
                continue
            r = seqs[f'QSP n={n}']
            ax.loglog(r[1], r[2] / T**2 + 1e-20,
                      color=color, lw=2.2, ls='-',
                      label=rf'QSP $n{{\!=\!}}{n}$', zorder=4)

        # Noise PSD overlay (scaled for visibility)
        w_vis = np.linspace(3e-2, 30, 2000)
        s_vis = S_func(w_vis)
        s_vis = s_vis / (np.max(s_vis) + 1e-30) * 0.25
        ax.fill_between(w_vis, 1e-20, s_vis + 1e-20,
                        color='0.85', zorder=0, label=r'$S(\omega)$ (scaled)')

        ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=11)
        ax.set_xlim([3e-2, 30])
        ax.set_ylim([1e-9, 1.5])
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

    axes[0].set_ylabel(r'$F(\omega)\,/\,T^2$', fontsize=11)

    # Single legend on first panel
    axes[0].legend(fontsize=7.5, loc='lower left', ncol=1)

    fig.suptitle(
        r'Three-level clock filter functions — $T = 2\pi$, all protocols at $\delta = 0$'
        '\n'
        r'Equiangular and QSP sequences optimised independently for each noise model',
        fontsize=11)

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'protocol_comparison.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')


if __name__ == '__main__':
    main()
