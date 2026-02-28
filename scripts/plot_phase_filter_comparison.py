"""
Phase-derivative filter function comparison: Ramsey, Rabi, and GPS protocols.

Two computation paths are shown for each protocol, distinguished by linestyle:

  Solid  — 3-level: fft_phase_filter on the full 3-level MultiLevelPulseSequence.
            phi(t) = conj(f)*g from the probe-subspace unitary; d phi/dt is FFT-ed.
  Dashed — 2-level: fft_phase_filter_2level on the qubit proxy sequence.
            Same phi(t) sampling but using the 2x2 qubit Cayley-Klein trajectory.

The result equals w^2 * Fe(w), emphasising sensitivity to frequency noise.

For instantaneous Ramsey phi(t) is piecewise constant, so d phi/dt ≈ 0 and
F_phase is negligible.  Continuous protocols (GPS, Rabi) show non-trivial spectra.

GPS qubit proxy: continuous_rabi_sequence (no |f> reference in 2-level model).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.core.multilevel import (
    MultiLevelPulseSequence,
    multilevel_ramsey,
)
from quantum_pulse_suite.core.three_level_filter import (
    fft_phase_filter,
    fft_phase_filter_2level,
)
from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.global_phase_spectroscopy import (
    GlobalPhaseSpectroscopySequence,
)
from quantum_pulse_suite import (
    ramsey_sequence,
    continuous_ramsey_sequence,
    continuous_rabi_sequence,
)

# =============================================================================
# Configuration
# =============================================================================

T_TOTAL    = 2 * np.pi
OMEGA_FAST = 20 * np.pi

FREQ_MIN = 0.05
FREQ_MAX = 50
N_FFT    = 2048

FIGURE_SIZE = (10, 6)
DPI = 150
OUTPUT_DIR  = 'figures/qubit_performance_plots'

M_Z = 0.0   # sigma_y measurement on clock


# =============================================================================
# Helpers
# =============================================================================

def _phase_ff_3L(seq):
    """3-level phase filter, masked to [FREQ_MIN, FREQ_MAX]."""
    freqs, F_phase, _, _ = fft_phase_filter(seq, n_samples=N_FFT, m_z=M_Z)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    return freqs[mask], F_phase[mask]


def _phase_ff_2L(qubit_seq):
    """2-level phase filter, masked to [FREQ_MIN, FREQ_MAX]."""
    freqs, F_phase = fft_phase_filter_2level(qubit_seq, n_samples=N_FFT)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    return freqs[mask], F_phase[mask]


def _add_pair(ax, color, label, f3L, Fp3L, f2L, Fp2L):
    """Plot 3-level (solid) and 2-level (dashed) for one protocol."""
    kw = dict(lw=2)
    ax.loglog(f3L, Fp3L / T_TOTAL**2 + 1e-20, color=color, ls='-',  **kw, label=label)
    ax.loglog(f2L, Fp2L / T_TOTAL**2 + 1e-20, color=color, ls='--', lw=1.4, alpha=0.85)


# =============================================================================
# Main
# =============================================================================

def main():
    fig_dir = Path(__file__).parent.parent / OUTPUT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    system = ThreeLevelClock()

    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T_TOTAL - 2 * tau_pi2

    # ── Instantaneous Ramsey ───────────────────────────────────────────────
    print("Instantaneous Ramsey")
    seq3 = multilevel_ramsey(system, system.probe, tau=T_TOTAL, delta=0.0)
    seq2 = ramsey_sequence(tau=T_TOTAL, delta=0.0)
    f3L, Fp3L = _phase_ff_3L(seq3)
    f2L, Fp2L = _phase_ff_2L(seq2)
    _add_pair(ax, 'C0', 'Ramsey (instant)', f3L, Fp3L, f2L, Fp2L)

    # ── Continuous Ramsey ──────────────────────────────────────────────────
    print("Continuous Ramsey")
    seq3 = MultiLevelPulseSequence(system, system.probe)
    seq3.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq3.add_free_evolution(tau_free, 0.0)
    seq3.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq2 = continuous_ramsey_sequence(omega=OMEGA_FAST, tau=T_TOTAL, delta=0.0)
    f3L, Fp3L = _phase_ff_3L(seq3)
    f2L, Fp2L = _phase_ff_2L(seq2)
    _add_pair(ax, 'C5', r'Ramsey (continuous $\pi/2$)', f3L, Fp3L, f2L, Fp2L)

    # ── Rabi m=1 ──────────────────────────────────────────────────────────
    print("Rabi m=1")
    omega_rabi = np.pi / T_TOTAL
    seq3 = MultiLevelPulseSequence(system, system.probe)
    seq3.add_continuous_pulse(omega_rabi, [1, 0, 0], 0.0, T_TOTAL)
    seq2 = continuous_rabi_sequence(omega=omega_rabi, tau=T_TOTAL, delta=0.0)
    f3L, Fp3L = _phase_ff_3L(seq3)
    f2L, Fp2L = _phase_ff_2L(seq2)
    _add_pair(ax, 'C1', 'Rabi (m=1)', f3L, Fp3L, f2L, Fp2L)

    # ── GPS m=1, 2, 8 ─────────────────────────────────────────────────────
    gps_cycles = [1, 2, 8]
    gps_colors = ['C2', 'C3', 'C4']

    for n_cyc, color in zip(gps_cycles, gps_colors):
        print(f"GPS m={n_cyc}")
        omega_gps = 2 * np.pi * n_cyc / T_TOTAL
        gps = GlobalPhaseSpectroscopySequence(
            system, n_cycles=n_cyc, omega=omega_gps, delta=0.0)
        seq2 = continuous_rabi_sequence(omega=omega_gps, tau=T_TOTAL, delta=0.0)
        f3L, Fp3L = _phase_ff_3L(gps._sequence)
        f2L, Fp2L = _phase_ff_2L(seq2)
        _add_pair(ax, color, rf'GPS (m={n_cyc})', f3L, Fp3L, f2L, Fp2L)

    # ── Axes ───────────────────────────────────────────────────────────────
    ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$F_{\rm phase}(\omega) / T^2$', fontsize=12)
    ax.set_title(
        r'Phase-derivative filter function $F_{\rm phase}(\omega) = \omega^2 F_e(\omega)$'
        '\n' + rf'$T = {T_TOTAL:.2f}$, FFT of $\dot{{\phi}}(t)$',
        fontsize=12)
    ax.set_xlim([FREQ_MIN, FREQ_MAX])
    ax.grid(True, alpha=0.3, which='both')

    # Protocol legend (colour, bottom-left)
    protocol_legend = ax.legend(fontsize=9, loc='lower left')

    # Method legend (linestyle, top-right)
    method_handles = [
        Line2D([0], [0], color='k', lw=2,   ls='-',  label='3-level'),
        Line2D([0], [0], color='k', lw=1.4, ls='--', alpha=0.85, label='2-level qubit'),
    ]
    ax.legend(handles=method_handles, fontsize=9, loc='upper right')
    ax.add_artist(protocol_legend)

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = fig_dir / f'phase_filter_comparison.{ext}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.close('all')


if __name__ == '__main__':
    main()
