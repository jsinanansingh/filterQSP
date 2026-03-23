"""
Filter function comparison: 2×2 grid, one panel per noise type + legend panel.

Panels: white (top-left) | 1/f (top-right)
        hp2 (bottom-left) | legend (bottom-right)

Each panel shows:
  - Ramsey (grey dashed) — reference
  - GPS m=1 (blue dash-dot) — reference
  - Double-π QSP n=2 (purple dotted) — DC sensitivity reference
  - Best equiangular for this noise type (orange)
  - Best QSP for this noise type (dark blue)
  - Noise PSD (grey fill, scaled for visibility)

Sequences are loaded from the FL-cutoff (ω_min=2π/T, ramsey_normalized) caches
that match Table I of the main text.

All protocols: T=2π, δ=0.
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
OMEGA_PLOT = np.logspace(-1, np.log10(30), 800)  # ω from 0.1 to 30
OMEGA_MIN  = 2 * np.pi / T                        # Fourier-limit cutoff

OMEGA_GPS1 = 2 * np.pi * 1 / T  # = 1.0

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'

# FL-cutoff caches (match Table I of the main text)
# equiangular: 20260320 run (more seeds, confirmed identical to 20260317)
# QSP: 20260320 definitive run (omega_max=4*omega_fast=251 rad/s, n=[4,8,13,16])
EQ_CACHE_PATH  = OUTPUT_DIR / 'equiangular_opt_cache_20260320_115752.npz'
QSP_CACHE_PATH = OUTPUT_DIR / 'qsp_opt_cache_20260320_170624.npz'

EQ_NS  = [4, 8, 16]
QSP_NS = [4, 8, 13]


def find_latest_equiangular_cache():
    """Return the equiangular cache path used by this script (for import by other scripts)."""
    return EQ_CACHE_PATH if EQ_CACHE_PATH.exists() else None

# Noise panels: (eq_cache_key, qsp_cache_key, panel_title, S_func)
# Keys must match those actually stored in the respective caches.
NOISE_PANELS = [
    ('white',      'White',           'White noise',              white_noise_psd()),
    ('1f',         '1ff',            r'$1/f$ noise',              one_over_f_psd()),
    ('highpass2',  'High-pass_w_c2', r'High-pass ($\omega_c=2$)', high_pass_psd(omega_c=2.0)),
]

# --- PRA-standard plot style -------------------------------------------------
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.size':           8,
    'axes.labelsize':      9,
    'axes.titlesize':      8,
    'xtick.labelsize':     7,
    'ytick.labelsize':     7,
    'legend.fontsize':     7,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '0.7',
    'lines.linewidth':     1.4,
    'axes.linewidth':      0.6,
    'xtick.major.width':   0.6,
    'ytick.major.width':   0.6,
    'xtick.minor.width':   0.4,
    'ytick.minor.width':   0.4,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
})

# Colour scheme
COLOR_RAMSEY   = '0.45'         # grey
COLOR_GPS1     = '#2166AC'      # steel blue
COLOR_DPI      = '#9B30FF'      # purple — double-π reference
COLOR_EQ_BEST  = '#D94F00'      # burnt orange
COLOR_QSP_BEST = '#0A3D6B'      # dark navy


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


def build_double_pi(system):
    return build_qsp_3level(system, T, 2, [np.pi, np.pi], [0.0, 0.0], OMEGA_FAST)


def build_equiangular(system, omega, phases):
    tau = T / len(phases)
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


def sigma_nu(res, S_func):
    sens_sq, _, _, freqs_fft, Fe_fft = res
    mask = freqs_fft >= OMEGA_MIN
    if np.count_nonzero(mask) < 2 or sens_sq < 1e-20:
        return np.inf
    nv = float(simpson(Fe_fft[mask] * S_func(freqs_fft[mask]),
                       x=freqs_fft[mask]) / (2 * np.pi))
    return nv / sens_sq


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    # ── Load caches ───────────────────────────────────────────────────────────
    if not EQ_CACHE_PATH.exists():
        raise FileNotFoundError(f'Equiangular cache not found: {EQ_CACHE_PATH}')
    eq_cache  = np.load(str(EQ_CACHE_PATH),  allow_pickle=True)
    qsp_cache = np.load(str(QSP_CACHE_PATH), allow_pickle=True) \
                if QSP_CACHE_PATH.exists() else None
    print(f'Equiangular cache: {EQ_CACHE_PATH.name}')
    print(f'QSP cache:         {QSP_CACHE_PATH.name if QSP_CACHE_PATH.exists() else "none"}')

    # ── Reference sequences (noise-independent) ───────────────────────────────
    print('\nBuilding reference sequences ...')
    res_ramsey = get_filter(build_ramsey(system))
    res_gps1   = get_filter(build_gps(system, OMEGA_GPS1))
    res_dpi    = get_filter(build_double_pi(system))

    # ── Per-noise sequences: pick best by σ²_ν ───────────────────────────────
    panel_data = []
    for eq_key, qsp_key, panel_title, S_func in NOISE_PANELS:
        # Best equiangular
        eq_best_res, eq_best_snu, eq_best_N = None, np.inf, None
        for N in EQ_NS:
            pk = f'eq_N{N}_{eq_key}'
            if f'{pk}_omega' not in eq_cache:
                continue
            omega  = float(eq_cache[f'{pk}_omega'])
            phases = np.asarray(eq_cache[f'{pk}_phases'], dtype=float)
            res    = get_filter(build_equiangular(system, omega, phases))
            snu    = sigma_nu(res, S_func)
            if snu < eq_best_snu:
                eq_best_snu, eq_best_res, eq_best_N = snu, res, N

        # Best QSP
        qsp_best_res, qsp_best_snu, qsp_best_n = None, np.inf, None
        if qsp_cache is not None:
            for n in QSP_NS:
                k = f'qsp_n{n}_{qsp_key}'
                if f'{k}_thetas' not in qsp_cache:
                    continue
                seq = build_qsp_3level(system, T, n,
                                       qsp_cache[f'{k}_thetas'],
                                       qsp_cache[f'{k}_phis'],
                                       float(qsp_cache[f'{k}_omega_fast']))
                res = get_filter(seq)
                snu = sigma_nu(res, S_func)
                if snu < qsp_best_snu:
                    qsp_best_snu, qsp_best_res, qsp_best_n = snu, res, n

        panel_data.append({
            'eq_res': eq_best_res,  'eq_N':  eq_best_N,
            'qsp_res': qsp_best_res, 'qsp_n': qsp_best_n,
            'S_func': S_func,       'title': panel_title,
        })
        print(f'  [{eq_key}]  best Eq N={eq_best_N} σ²_ν={eq_best_snu:.3e}'
              f'  |  best QSP n={qsp_best_n} σ²_ν={qsp_best_snu:.3e}')

    # ── Figure: 2×2 grid (3 noise panels + legend panel) ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4),
                             constrained_layout=True)

    panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    legend_ax  = axes[1, 1]

    for ax, (_, _, _, S_func), pd in zip(panel_axes, NOISE_PANELS, panel_data):

        # Reference curves
        l_ramsey, = ax.loglog(res_ramsey[1], res_ramsey[2] / T**2 + 1e-20,
                              color=COLOR_RAMSEY, lw=1.8, ls='--', zorder=3)
        l_gps1, = ax.loglog(res_gps1[1], res_gps1[2] / T**2 + 1e-20,
                            color=COLOR_GPS1, lw=1.8, ls='-.', zorder=3)
        l_dpi, = ax.loglog(res_dpi[1], res_dpi[2] / T**2 + 1e-20,
                           color=COLOR_DPI, lw=1.5, ls=':', zorder=3)

        # Best equiangular
        l_eq = None
        if pd['eq_res'] is not None:
            r = pd['eq_res']
            l_eq, = ax.loglog(r[1], r[2] / T**2 + 1e-20,
                              color=COLOR_EQ_BEST, lw=2.0, ls='-', zorder=4)

        # Best QSP
        l_qsp = None
        if pd['qsp_res'] is not None:
            r = pd['qsp_res']
            l_qsp, = ax.loglog(r[1], r[2] / T**2 + 1e-20,
                               color=COLOR_QSP_BEST, lw=2.0, ls='-', zorder=4)

        # Noise PSD fill
        w_vis = np.linspace(1e-1, 30, 2000)
        s_vis = S_func(w_vis)
        s_vis = s_vis / (np.max(s_vis) + 1e-30) * 0.12
        ax.fill_between(w_vis, 1e-20, s_vis + 1e-20, color='0.88', zorder=0)

        ax.set_xlim([1e-1, 30])
        ax.set_ylim([1e-8, 1.5])
        ax.set_title(pd['title'])
        ax.grid(True, alpha=0.25, which='both')
        ax.tick_params(which='both', top=True, right=True)

    # Shared axis labels
    axes[0, 0].set_ylabel(r'$\mathcal{F}_e(\omega)\,/\,T^2$')
    axes[1, 0].set_ylabel(r'$\mathcal{F}_e(\omega)\,/\,T^2$')
    axes[1, 0].set_xlabel(r'$\omega$ (rad s$^{-1}$)')
    axes[1, 1].set_xlabel(r'$\omega$ (rad s$^{-1}$)')

    # ── Legend panel ──────────────────────────────────────────────────────────
    legend_ax.axis('off')
    # Collect legend handles from the last active panel
    ax_last = panel_axes[-1]
    pd_last = panel_data[-1]

    legend_entries = [
        (matplotlib.lines.Line2D([0], [0], color=COLOR_RAMSEY,   lw=1.8, ls='--'), 'Ramsey'),
        (matplotlib.lines.Line2D([0], [0], color=COLOR_GPS1,     lw=1.8, ls='-.'), r'GPS $m{=}1$'),
        (matplotlib.lines.Line2D([0], [0], color=COLOR_DPI,      lw=1.5, ls=':' ), r'QSP $n{=}2$ (double-$\pi$)'),
        (matplotlib.lines.Line2D([0], [0], color=COLOR_EQ_BEST,  lw=2.0, ls='-' ), r'Best equiangular (noise-opt.)'),
        (matplotlib.lines.Line2D([0], [0], color=COLOR_QSP_BEST, lw=2.0, ls='-' ), r'Best QSP (noise-opt.)'),
        (matplotlib.patches.Patch(facecolor='0.88', edgecolor='0.7'), r'$S(\omega)$ (scaled)'),
    ]
    handles, labels = zip(*legend_entries)
    legend_ax.legend(handles, labels,
                     loc='center',
                     frameon=True, framealpha=0.9,
                     edgecolor='0.7', handlelength=2.5,
                     title=r'$T=2\pi,\;\delta=0$',
                     title_fontsize=8)

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'protocol_comparison.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')


if __name__ == '__main__':
    main()
