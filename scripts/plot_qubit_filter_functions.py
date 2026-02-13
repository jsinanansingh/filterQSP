"""
Filter function Bloch vector components |Fx|^2, |Fy|^2, |Fz|^2
for pulse sequences.

Produces two separate figures:
  - CPMG: Ramsey, CPMG-1, CPMG-4, CPMG-8 (instantaneous vs continuous)
  - GPS:  GPS n=1, GPS n=2, GPS n=8 (continuous only — no instantaneous analog)

Usage:
    python scripts/plot_qubit_filter_functions.py

Output:
    figures/qubit_filter_functions/cpmg_filter_functions.{pdf,png}
    figures/qubit_filter_functions/gps_filter_functions.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite import (
    ramsey_sequence,
    cpmg_sequence,
    continuous_ramsey_sequence,
    continuous_cpmg_sequence,
    ContinuousPulseSequence,
)

# =============================================================================
# Configuration
# =============================================================================

TAU = 2 * np.pi
DELTA = 0.25
OMEGA_FAST = 20 * np.pi

FREQ_MIN = 0.01
FREQ_MAX = 50
N_FREQ = 2000

DPI = 150
FONT_SIZE = 11
OUTPUT_DIR = 'figures/qubit_filter_functions'

C_FX = '#e41a1c'
C_FY = '#377eb8'
C_FZ = '#4daf4a'

CPMG_LEGEND_ELEMENTS = [
    Line2D([0], [0], color=C_FX, linewidth=2, label='$|F_x|^2$'),
    Line2D([0], [0], color=C_FY, linewidth=2, label='$|F_y|^2$'),
    Line2D([0], [0], color=C_FZ, linewidth=2, label='$|F_z|^2$'),
    Line2D([0], [0], color='gray', linewidth=2, label='Continuous'),
    Line2D([0], [0], color='gray', linewidth=2, linestyle='--',
           label='Instantaneous'),
]

GPS_LEGEND_ELEMENTS = [
    Line2D([0], [0], color=C_FX, linewidth=2, label='$|F_x|^2$'),
    Line2D([0], [0], color=C_FY, linewidth=2, label='$|F_y|^2$'),
    Line2D([0], [0], color=C_FZ, linewidth=2, label='$|F_z|^2$'),
]

# =============================================================================
# Builders
# =============================================================================


def build_gps_qubit(n_cycles):
    """Qubit GPS: fast pi/2 + continuous Rabi drive (n cycles) + fast pi/2."""
    omega_drive = 2 * np.pi * n_cycles / TAU
    tau_pi2 = np.pi / (2 * OMEGA_FAST)
    seq = ContinuousPulseSequence()
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], DELTA, tau_pi2)
    seq.add_continuous_pulse(omega_drive, [1, 0, 0], DELTA, TAU)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], DELTA, tau_pi2)
    seq.compute_polynomials()
    return seq


def plot_filter_panel(ax, seq_inst, seq_cont, frequencies, title):
    """Plot |Fx|^2, |Fy|^2, |Fz|^2 for both sequences on one axis."""
    ff_i = seq_inst.get_filter_function_calculator()
    Fx_i, Fy_i, Fz_i = ff_i.filter_function(frequencies)

    ff_c = seq_cont.get_filter_function_calculator()
    Fx_c, Fy_c, Fz_c = ff_c.filter_function(frequencies)

    for Fj_i, Fj_c, color in [
        (Fx_i, Fx_c, C_FX),
        (Fy_i, Fy_c, C_FY),
        (Fz_i, Fz_c, C_FZ),
    ]:
        ax.plot(frequencies, np.abs(Fj_c)**2, color=color,
                linewidth=1.5, alpha=0.9, zorder=3)
        ax.plot(frequencies, np.abs(Fj_i)**2, color=color,
                linewidth=1.5, alpha=0.55, linestyle='--', zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\omega$', fontsize=11)
    ax.set_ylabel(r'$|F_j(\omega)|^2$', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.grid(True, alpha=0.12, which='both')
    ymin = max(1e-10, ax.get_ylim()[0])
    ax.set_ylim(bottom=ymin)


def plot_gps_panel(ax, seq_gps, frequencies, title):
    """Plot |Fx|^2, |Fy|^2, |Fz|^2 for a single GPS sequence (continuous only)."""
    ff = seq_gps.get_filter_function_calculator()
    Fx, Fy, Fz = ff.filter_function(frequencies)

    for Fj, color, label in [
        (Fx, C_FX, '$|F_x|^2$'),
        (Fy, C_FY, '$|F_y|^2$'),
        (Fz, C_FZ, '$|F_z|^2$'),
    ]:
        ax.plot(frequencies, np.abs(Fj)**2, color=color,
                linewidth=1.5, alpha=0.9, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\omega$', fontsize=11)
    ax.set_ylabel(r'$|F_j(\omega)|^2$', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.grid(True, alpha=0.12, which='both')
    ymin = max(1e-10, ax.get_ylim()[0])
    ax.set_ylim(bottom=ymin)


# =============================================================================
# Main
# =============================================================================


def main():
    rcParams.update({
        'font.size': FONT_SIZE,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })

    out_dir = Path(__file__).parent.parent / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)

    # ===================== CPMG figure (2x2) =====================
    fig_c, axes_c = plt.subplots(2, 2, figsize=(12, 9))
    axes_c = axes_c.flatten()

    cpmg_protos = []
    # Ramsey
    si = ramsey_sequence(tau=TAU, delta=DELTA)
    si.compute_polynomials()
    sc = continuous_ramsey_sequence(omega=OMEGA_FAST, tau=TAU, delta=DELTA)
    sc.compute_polynomials()
    cpmg_protos.append(('Ramsey', si, sc))

    # CPMG-1, 4, 8
    for n in [1, 4, 8]:
        si = cpmg_sequence(tau=TAU, n_pulses=n, delta=DELTA)
        si.compute_polynomials()
        sc = continuous_cpmg_sequence(
            omega=OMEGA_FAST, tau=TAU, n_pulses=n, delta=DELTA)
        sc.compute_polynomials()
        cpmg_protos.append((f'CPMG-{n}', si, sc))

    for idx, (name, si, sc) in enumerate(cpmg_protos):
        plot_filter_panel(axes_c[idx], si, sc, frequencies, name)

    axes_c[0].legend(handles=CPMG_LEGEND_ELEMENTS, fontsize=8, loc='lower left',
                     framealpha=0.9)

    fig_c.suptitle(
        'CPMG Filter Functions: Instantaneous vs Continuous\n'
        f'$\\tau = {TAU:.2f}$, $\\delta = {DELTA}$, '
        f'$\\Omega = {OMEGA_FAST:.1f}$',
        fontsize=13, y=1.01)
    fig_c.tight_layout()

    for ext in ['pdf', 'png']:
        p = out_dir / f'cpmg_filter_functions.{ext}'
        fig_c.savefig(p, dpi=DPI, bbox_inches='tight')
        print(f'Saved: {p}')

    # ===================== GPS figure (2x2, 4th = legend) =====================
    # GPS is inherently continuous (laser on during entire interrogation),
    # so there is no instantaneous counterpart to compare against.
    fig_g, axes_g = plt.subplots(2, 2, figsize=(12, 9))
    axes_g = axes_g.flatten()

    for idx, n in enumerate([1, 2, 8]):
        seq_gps = build_gps_qubit(n)
        omega_d = 2 * np.pi * n / TAU
        plot_gps_panel(
            axes_g[idx], seq_gps, frequencies,
            f'GPS $n={n}$  ($\\Omega_d={omega_d:.1f}$)')

    # Legend panel
    ax_leg = axes_g[3]
    ax_leg.axis('off')
    ax_leg.legend(handles=GPS_LEGEND_ELEMENTS, loc='center', fontsize=13,
                  frameon=True, framealpha=0.9, edgecolor='gray')
    ax_leg.set_title('Legend', fontsize=12, fontweight='bold')

    fig_g.suptitle(
        'GPS Filter Functions (Continuous)\n'
        f'$\\tau = {TAU:.2f}$, $\\delta = {DELTA}$',
        fontsize=13, y=1.01)
    fig_g.tight_layout()

    for ext in ['pdf', 'png']:
        p = out_dir / f'gps_filter_functions.{ext}'
        fig_g.savefig(p, dpi=DPI, bbox_inches='tight')
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
