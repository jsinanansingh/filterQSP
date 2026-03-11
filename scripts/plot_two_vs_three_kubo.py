"""
Compare Kubo filter functions for 2-level (qubit) vs 3-level (Lambda clock)
for Ramsey, GPS m=1, and GPS m=8.

2-level: H_noise = delta * sigma_z / 2, noise enters as phase noise on the qubit.
         Filter function F_simple(omega) = |(m_hat x F_tilde(omega)) . r0|^2
         where F_tilde is the FT of the toggling-frame Bloch vector of sigma_z/2.

3-level: H_noise = delta * |e><e|, noise shifts only the excited state.
         Filter function Fe(omega) from analytic_filter (m_y component).

Both evaluate at T = 2*pi with GPS m=1 (Omega=1) and GPS m=8 (Omega=8).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
from quantum_pulse_suite.core.three_level_filter import (
    kubo_filter_2level_full_analytic,
    analytic_filter,
)
from quantum_pulse_suite import (
    continuous_ramsey_sequence,
    continuous_rabi_sequence,
)

# =============================================================================
# Parameters
# =============================================================================

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi   # fast pi/2 pulses for Ramsey

OMEGA_GPS1 = 2 * np.pi * 1 / T   # = 1.0 rad/s
OMEGA_GPS8 = 2 * np.pi * 8 / T   # = 8.0 rad/s

FREQS = np.logspace(-1.5, np.log10(30), 600)

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'


# =============================================================================
# Build sequences
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

    # ── Build sequences ───────────────────────────────────────────────────────
    # 3-level
    s3_ramsey = build_3level_ramsey(system)
    s3_gps1   = build_3level_gps(system, OMEGA_GPS1)
    s3_gps8   = build_3level_gps(system, OMEGA_GPS8)

    # 2-level
    tau_pi2   = np.pi / (2 * OMEGA_FAST)
    tau_free  = T - 2 * tau_pi2
    s2_ramsey = continuous_ramsey_sequence(omega=OMEGA_FAST, tau=T, delta=0.0)
    s2_gps1   = continuous_rabi_sequence(omega=OMEGA_GPS1, tau=T, delta=0.0)
    s2_gps8   = continuous_rabi_sequence(omega=OMEGA_GPS8, tau=T, delta=0.0)

    # ── Compute filter functions ──────────────────────────────────────────────
    print('Computing 3-level filter functions ...')
    _, Fe3_ramsey = analytic_filter(s3_ramsey, FREQS, m_y=1.0)
    _, Fe3_gps1   = analytic_filter(s3_gps1,   FREQS, m_y=1.0)
    _, Fe3_gps8   = analytic_filter(s3_gps8,   FREQS, m_y=1.0)

    print('Computing 2-level filter functions ...')
    _, F2_ramsey_simple, _ = kubo_filter_2level_full_analytic(
        s2_ramsey, FREQS, m_hat=[0, 1, 0], r0=[1, 0, 0])
    _, F2_gps1_simple, _   = kubo_filter_2level_full_analytic(
        s2_gps1,   FREQS, m_hat=[0, 1, 0], r0=[1, 0, 0])
    _, F2_gps8_simple, _   = kubo_filter_2level_full_analytic(
        s2_gps8,   FREQS, m_hat=[0, 1, 0], r0=[1, 0, 0])

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    eps = 1e-20   # floor to avoid log(0)

    # Ramsey: red
    ax.loglog(FREQS, Fe3_ramsey / T**2 + eps,       color='C3', lw=2)
    ax.loglog(FREQS, F2_ramsey_simple / T**2 + eps, color='C3', lw=2, ls='--')

    # GPS m=1: blue
    ax.loglog(FREQS, Fe3_gps1 / T**2 + eps,         color='C0', lw=2)
    ax.loglog(FREQS, F2_gps1_simple / T**2 + eps,   color='C0', lw=2, ls='--')

    # GPS m=8: green
    ax.loglog(FREQS, Fe3_gps8 / T**2 + eps,         color='C2', lw=2)
    ax.loglog(FREQS, F2_gps8_simple / T**2 + eps,   color='C2', lw=2, ls='--')

    # Harmonic markers
    for k in range(1, 10):
        if OMEGA_GPS1 * k <= 30:
            ax.axvline(OMEGA_GPS1 * k, color='C0', lw=0.5, ls=':', alpha=0.35)
    for k in range(1, 5):
        if OMEGA_GPS8 * k <= 30:
            ax.axvline(OMEGA_GPS8 * k, color='C2', lw=0.5, ls=':', alpha=0.35)

    ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=12)
    ax.set_ylabel(r'$F_e(\omega)\,/\,T^2$', fontsize=12)
    ax.set_title(
        r'Kubo filter functions: 2-level (qubit, $H_\delta{=}\frac{\delta}{2}\sigma_z$)'
        r' vs 3-level ($\Lambda$, $H_\delta{=}\delta|e\rangle\langle e|$)'
        r'  |  $T = 2\pi$',
        fontsize=10)
    ax.set_xlim([FREQS[0], 30])
    ax.set_ylim([1e-9, 1.5])
    ax.grid(True, alpha=0.3, which='both')

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color='C3', lw=2, label='Ramsey'),
        mlines.Line2D([], [], color='C0', lw=2, label=r'GPS $m{=}1$'),
        mlines.Line2D([], [], color='C2', lw=2, label=r'GPS $m{=}8$'),
        mlines.Line2D([], [], color='k',  lw=2,        ls='-',  label=r'3-level $(\Lambda)$'),
        mlines.Line2D([], [], color='k',  lw=2,        ls='--', label='2-level (qubit)'),
    ]
    ax.legend(handles=handles, fontsize=9, loc='lower left')
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'two_vs_three_kubo.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')


if __name__ == '__main__':
    main()
