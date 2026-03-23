"""
Compare Kubo filter functions for 2-level (qubit) vs 3-level (Lambda clock).

2-level (optimal Ramsey):
    H_noise = beta * |e><e|,  M = sigma_y,  initial state = (|g> + |m>) / sqrt(2).
    This is the optimal measurement basis for optical Ramsey; it gives
        F_e^(2L)(omega) = T^2 * sinc^2(omega * T / (2*pi))
    with F_e^(2L)(0) = T^2 = (2*pi)^2.

3-level Lambda clock:
    H_noise = delta * |e><e|.
    Ramsey:      F_e(0) = T^2 / 4  (4x worse because |m> is reference, not signal).
    GPS m=1,8:   computed numerically via analytic_filter.
    Double-pi QSP n=3: thetas ~ [pi, 0, pi], achieves F_e(0) ~ T^2 (matches 2-level!).
      Parameters loaded from the latest QSP optimisation cache.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
from quantum_pulse_suite.core.three_level_filter import analytic_filter
from quantum_pulse_suite.analysis.pulse_optimizer import build_qsp_3level

# =============================================================================
# Parameters
# =============================================================================

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi   # fast pi/2 (and pi) pulses

OMEGA_GPS1 = 2 * np.pi * 1 / T   # = 1.0 rad/s
OMEGA_GPS8 = 2 * np.pi * 8 / T   # = 8.0 rad/s

FREQS = np.logspace(-2, np.log10(30), 600)

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'

# =============================================================================
# Helpers
# =============================================================================

def build_3level_ramsey(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


def build_3level_gps(system, omega):
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, T)
    seq.compute_polynomials()
    return seq


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    # ── Double-pi QSP n=2: fixed reference, no cache needed ──────────────────
    thetas_dpi = np.array([np.pi, np.pi])
    phis_dpi   = np.array([0.0,   0.0])
    print(f'Double-pi QSP n=2: thetas={thetas_dpi},  phis={phis_dpi}')
    print(f'  Expected F_e(0) ~ T^2 = {T**2:.4f}')

    # ── Build 3-level sequences ───────────────────────────────────────────────
    s3_ramsey = build_3level_ramsey(system)
    s3_gps1   = build_3level_gps(system, OMEGA_GPS1)
    s3_gps8   = build_3level_gps(system, OMEGA_GPS8)
    s3_dpi    = build_qsp_3level(system, T, n=2,
                                 thetas=thetas_dpi, phis=phis_dpi,
                                 omega_fast=OMEGA_FAST)

    # ── Compute 3-level filter functions (analytic) ───────────────────────────
    print('Computing 3-level filter functions ...')
    _, Fe3_ramsey = analytic_filter(s3_ramsey, FREQS, m_y=1.0)
    _, Fe3_gps1   = analytic_filter(s3_gps1,   FREQS, m_y=1.0)
    _, Fe3_gps8   = analytic_filter(s3_gps8,   FREQS, m_y=1.0)
    _, Fe3_dpi    = analytic_filter(s3_dpi,     FREQS, m_y=1.0)

    # ── 2-level Ramsey (analytic): F_e(omega) = T^2 * sinc^2(omega*T/(2*pi)) ─
    # numpy sinc(x) = sin(pi*x)/(pi*x), so sinc(omega*T/(2*pi)) = sin(omega*T/2)/(omega*T/2)
    Fe2_ramsey = T**2 * np.sinc(FREQS * T / (2 * np.pi))**2
    print(f'2-level Ramsey  F_e(0) ~ {T**2:.4f}  (T^2 = {T**2:.4f})')
    print(f'3-level Ramsey  F_e~DC = {Fe3_ramsey[0]:.4f}  (T^2/4 = {T**2/4:.4f})')
    print(f'Double-pi QSP   F_e~DC = {Fe3_dpi[0]:.4f}')

    # ── PRA-standard plot style ───────────────────────────────────────────────
    matplotlib.rcParams.update({
        'font.family':       'serif',
        'font.size':          8,
        'axes.labelsize':     9,
        'axes.titlesize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'legend.framealpha': 0.9,
        'legend.edgecolor':  '0.7',
        'lines.linewidth':    1.4,
        'axes.linewidth':     0.6,
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'xtick.minor.width':  0.4,
        'ytick.minor.width':  0.4,
        'xtick.direction':   'in',
        'ytick.direction':   'in',
    })

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.375, 2.8))
    eps = 1e-20

    # 3-level GPS m=1 (blue) and m=8 (green) — background
    ax.loglog(FREQS, Fe3_gps1 / T**2 + eps,    color='C0', lw=1.6, ls='-',  zorder=2)
    ax.loglog(FREQS, Fe3_gps8 / T**2 + eps,    color='C2', lw=1.6, ls='-',  zorder=2)

    # 3-level Ramsey (red dashed) — foreground
    ax.loglog(FREQS, Fe3_ramsey / T**2 + eps,  color='C3', lw=2.0, ls='--', zorder=4)

    # 2-level Ramsey = 3-level double-pi QSP (orange dashed) — foreground, single curve
    ax.loglog(FREQS, Fe2_ramsey / T**2 + eps,  color='C1', lw=2.0, ls='--', zorder=5)

    # Harmonic markers
    for k in range(1, 10):
        if OMEGA_GPS1 * k <= 30:
            ax.axvline(OMEGA_GPS1 * k, color='C0', lw=0.5, ls=':', alpha=0.35)
    for k in range(1, 5):
        if OMEGA_GPS8 * k <= 30:
            ax.axvline(OMEGA_GPS8 * k, color='C2', lw=0.5, ls=':', alpha=0.35)

    ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)')
    ax.set_ylabel(r'$\mathcal{F}_e(\omega)\,/\,T^2$')
    ax.set_xlim([0.01, 30])
    ax.set_ylim([1e-9, 1.5])
    ax.grid(True, alpha=0.25, which='both')
    ax.tick_params(which='both', top=True, right=True)

    handles = [
        mlines.Line2D([], [], color='C1', lw=2.0, ls='--',
                      label=r'2-level Ramsey / 3-level $\pi$-QSP'),
        mlines.Line2D([], [], color='C3', lw=2.0, ls='--',
                      label=r'3-level Ramsey'),
        mlines.Line2D([], [], color='C0', lw=1.6, ls='-',
                      label=r'GPS $m{=}1$'),
        mlines.Line2D([], [], color='C2', lw=1.6, ls='-',
                      label=r'GPS $m{=}8$'),
    ]
    ax.legend(handles=handles, loc='lower left',
              frameon=True, handlelength=1.5, fontsize=7)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'two_vs_three_kubo.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')


if __name__ == '__main__':
    main()
