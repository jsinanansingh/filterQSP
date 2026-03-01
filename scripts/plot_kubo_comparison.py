"""
Kubo variance filter function comparison: Ramsey, Rabi, and GPS protocols.

Three separate figures:

  Figure 1 (3-level):  kubo_filter_3level_analytic
    r(ω) = ∫ ⟨ψ₀|[M, H̃(t)]|ψ₀⟩ e^{-iωt} dt  evaluated analytically
    using QSP polynomial segments.  H̃(t) = U†(t)|e⟩⟨e|U(t).
    ψ₀ = (|g⟩+|m⟩)/√2.  M = σ_y^{gm}.

  Figure 2 (2-level):  kubo_filter_2level_analytic
    r(ω) = ∫ (m̂ × R(t))·r₀ e^{-iωt} dt  evaluated analytically
    using QSP polynomial segments.
    m̂ = (0,1,0) = σ_y, r₀ = (1,0,0) for equal superposition.
    GPS 2-level proxy: continuous_rabi_sequence (no |m⟩ reference).

  Figure 3 (comparison):  kubo_filter_3level_analytic  vs  analytic_three_level_filter
    Overlays per protocol:
      solid  — full Fe(ω) from analytic_three_level_filter (m_y=1, m_x=m_z=0)
               = ½|Φ(ω)|² + |Chi(ω)|²
      dashed — simplified Kubo  m_y²|Chi(ω)|²  (Chi term only)
    Difference reveals the ½|Φ|² contribution absent from the Kubo approximation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.core.multilevel import (
    MultiLevelPulseSequence,
)
from quantum_pulse_suite.core.three_level_filter import (
    kubo_filter_2level_analytic,
    kubo_filter_3level_analytic,
    analytic_three_level_filter,
)
from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.global_phase_spectroscopy import (
    GlobalPhaseSpectroscopySequence,
)
from quantum_pulse_suite import (
    continuous_ramsey_sequence,
    continuous_rabi_sequence,
)

# =============================================================================
# Configuration
# =============================================================================

T_TOTAL    = 2 * np.pi
OMEGA_FAST = 20 * np.pi

FREQ_MIN = 0.1
FREQ_MAX = 50
N_FREQ   = 500

FIGURE_SIZE = (8, 5)
DPI = 150
OUTPUT_DIR  = 'figures/qubit_performance_plots'

M_HAT = np.array([0., 1., 0.])
R0    = np.array([1., 0., 0.])

PROTOCOLS = [
    ('C5', r'Ramsey (cont. $\pi/2$)'),
    ('C1', 'Rabi (m=1)'),
    ('C2', 'GPS (m=1)'),
    ('C3', 'GPS (m=2)'),
    ('C4', 'GPS (m=8)'),
]


# =============================================================================
# Build sequences
# =============================================================================

def build_sequences(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T_TOTAL - 2 * tau_pi2

    seqs3, seqs2 = [], []

    # Continuous Ramsey
    s3 = MultiLevelPulseSequence(system, system.probe)
    s3.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    s3.add_free_evolution(tau_free, 0.0)
    s3.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    s3.compute_polynomials()
    s2 = continuous_ramsey_sequence(omega=OMEGA_FAST, tau=T_TOTAL, delta=0.0)
    seqs3.append(s3); seqs2.append(s2)

    # Rabi m=1
    omega_rabi = np.pi / T_TOTAL
    s3 = MultiLevelPulseSequence(system, system.probe)
    s3.add_continuous_pulse(omega_rabi, [1, 0, 0], 0.0, T_TOTAL)
    s3.compute_polynomials()
    s2 = continuous_rabi_sequence(omega=omega_rabi, tau=T_TOTAL, delta=0.0)
    seqs3.append(s3); seqs2.append(s2)

    # GPS m=1, 2, 8
    for n_cyc in [1, 2, 8]:
        omega_gps = 2 * np.pi * n_cyc / T_TOTAL
        gps = GlobalPhaseSpectroscopySequence(
            system, n_cycles=n_cyc, omega=omega_gps, delta=0.0)
        gps._sequence.compute_polynomials()
        s2 = continuous_rabi_sequence(omega=omega_gps, tau=T_TOTAL, delta=0.0)
        seqs3.append(gps._sequence); seqs2.append(s2)

    return seqs3, seqs2


# =============================================================================
# Plot helpers
# =============================================================================

def _decorate(ax, title):
    ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$F_{\rm Kubo}(\omega) / T^2$', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.set_xlim([FREQ_MIN, FREQ_MAX])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='lower left')


# =============================================================================
# Main
# =============================================================================

def main():
    fig_dir = Path(__file__).parent.parent / OUTPUT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)
    system = ThreeLevelClock()

    print("Building sequences...")
    seqs3, seqs2 = build_sequences(system)

    # ── Figure 1: 3-level ──────────────────────────────────────────────────
    print("Computing 3-level Kubo filter functions...")
    fig3, ax3 = plt.subplots(figsize=FIGURE_SIZE)

    for (color, label), s3 in zip(PROTOCOLS, seqs3):
        print(f"  {label}")
        _, Fk = kubo_filter_3level_analytic(s3, frequencies)
        ax3.loglog(frequencies, Fk / T_TOTAL**2 + 1e-20, color=color, lw=2, label=label)

    _decorate(ax3,
        r'Kubo filter function: 3-level clock, $M = \sigma_y^{gm}$'
        '\n'
        r'$r(t) = \langle(|g\rangle+|m\rangle)/\sqrt{2}\,|\,[\sigma_y^{gm},\,\tilde{H}(t)]\,'
        r'|\,(|g\rangle+|m\rangle)/\sqrt{2}\rangle$')
    fig3.tight_layout()

    for ext in ['pdf', 'png']:
        path = fig_dir / f'kubo_3level.{ext}'
        fig3.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {path}")

    # ── Figure 2: 2-level ──────────────────────────────────────────────────
    print("Computing 2-level Kubo filter functions...")
    fig2, ax2 = plt.subplots(figsize=FIGURE_SIZE)

    for (color, label), s2 in zip(PROTOCOLS, seqs2):
        print(f"  {label}")
        _, Fk = kubo_filter_2level_analytic(s2, frequencies, m_hat=M_HAT, r0=R0)
        ax2.loglog(frequencies, Fk / T_TOTAL**2 + 1e-20, color=color, lw=2, label=label)

    _decorate(ax2,
        r'Kubo filter function: 2-level qubit, $M = \sigma_y$'
        '\n'
        r'$r(t) = (\hat{m}\times R(t))\cdot r_0$, '
        r'$\hat{m}=(0,1,0)$, $r_0=(1,0,0)$')
    fig2.tight_layout()

    for ext in ['pdf', 'png']:
        path = fig_dir / f'kubo_2level.{ext}'
        fig2.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {path}")

    # ── Figure 3: Kubo vs full Fe comparison ──────────────────────────────
    print("Computing Kubo vs full Fe comparison...")
    fig_cmp, ax_cmp = plt.subplots(figsize=FIGURE_SIZE)

    for (color, label), s3 in zip(PROTOCOLS, seqs3):
        print(f"  {label}")
        _, Fk = kubo_filter_3level_analytic(s3, frequencies)
        _, Fe, _, _ = analytic_three_level_filter(s3, frequencies, m_y=1.0)
        ax_cmp.loglog(frequencies, Fe / T_TOTAL**2 + 1e-20,
                      color=color, lw=2, ls='-',  label=label)
        ax_cmp.loglog(frequencies, Fk / T_TOTAL**2 + 1e-20,
                      color=color, lw=1.5, ls='--')

    # Legend: protocol colours (solid proxy) + linestyle guide
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='k', lw=2,   ls='-',  label=r'Full $F_e$ ($\frac{1}{2}|\Phi|^2 + |\chi|^2$)'),
        Line2D([0], [0], color='k', lw=1.5, ls='--', label=r'Kubo $m_y^2|\chi|^2$'),
    ]
    proto_legend = ax_cmp.legend(fontsize=9, loc='lower left')
    ax_cmp.add_artist(proto_legend)
    ax_cmp.legend(handles=style_handles, fontsize=9, loc='upper right')

    ax_cmp.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax_cmp.set_ylabel(r'$F(\omega) / T^2$', fontsize=12)
    ax_cmp.set_title(
        r'Full $F_e$ (solid) vs Kubo $m_y^2|\chi|^2$ (dashed), $m_y=1$'
        '\n'
        r'Difference $= \frac{1}{2}|\Phi(\omega)|^2$', fontsize=11)
    ax_cmp.set_xlim([FREQ_MIN, FREQ_MAX])
    ax_cmp.grid(True, alpha=0.3, which='both')
    fig_cmp.tight_layout()

    for ext in ['pdf', 'png']:
        path = fig_dir / f'kubo_filter_comparison.{ext}'
        fig_cmp.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.close('all')


if __name__ == '__main__':
    main()
