"""
Plot comparing continuous vs instantaneous CPMG sequences.

This script compares filter functions for CPMG sequences where the π-pulses
are either:
- Instantaneous (ideal delta-function pulses)
- Continuous (finite duration with Rabi frequency Ω)

For each case, we plot N = 1, 4, 8 π-pulses with:
- Same color for each N value
- Solid lines for instantaneous
- Dotted lines for continuous

Usage:
    python scripts/plot_cpmg_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite import (
    InstantaneousPulseSequence,
    ContinuousPulseSequence,
    cpmg_sequence,
)

# =============================================================================
# Plot Configuration - EDIT THESE PARAMETERS
# =============================================================================

# Sequence parameters
TAU_TOTAL = 1.0          # Total free evolution time
DELTA = 0.0              # Detuning during free evolution
N_PULSES_LIST = [1, 4, 8]  # Number of π-pulses to compare

# Continuous pulse parameters
OMEGA_PI = 10 * np.pi    # Rabi frequency for π-pulses (high = fast pulses)
TAU_PI = np.pi / OMEGA_PI  # Duration of each π-pulse

# Frequency range for filter function
FREQ_MIN = 0.1
FREQ_MAX = 100
N_FREQ_POINTS = 500

# Plot styling
FIGURE_SIZE = (10, 7)
DPI = 150
FONT_SIZE = 12
LINE_WIDTH = 2.0
USE_LOG_SCALE = True

# Colors for different N values
COLORS = {
    1: '#1f77b4',   # Blue
    4: '#2ca02c',   # Green
    8: '#d62728',   # Red
}

# Line styles
STYLE_INSTANTANEOUS = '-'   # Solid
STYLE_CONTINUOUS = '--'     # Dashed

# Output
SAVE_FIGURE = True
OUTPUT_PATH = 'figures/cpmg_instant_vs_continuous.pdf'

# =============================================================================
# Build Sequences
# =============================================================================

def build_instantaneous_cpmg(tau_total, n_pulses, delta=0.0):
    """
    Build instantaneous CPMG sequence using the library function.

    Structure: π/2_x - (τ/2N - π_y - τ/2N)×N - π/2_x
    """
    return cpmg_sequence(tau=tau_total, n_pulses=n_pulses, delta=delta)


def build_continuous_cpmg(tau_total, n_pulses, omega_pi, delta=0.0):
    """
    Build continuous CPMG sequence with finite-duration π-pulses.

    Structure: π/2_x - (τ/2N - π_y - τ/2N)×N - π/2_x

    The π-pulses have Rabi frequency omega_pi and duration π/omega_pi.
    """
    if n_pulses < 1:
        raise ValueError("n_pulses must be at least 1")

    tau_pi = np.pi / omega_pi   # Duration of π-pulse
    tau_pi_half = (np.pi / 2) / omega_pi  # Duration of π/2-pulse

    # Free evolution interval between pulses
    interval = tau_total / (2 * n_pulses)

    seq = ContinuousPulseSequence()

    # Initial π/2 pulse on x-axis
    seq.add_continuous_pulse(omega=omega_pi, axis=[1, 0, 0], delta=0, tau=tau_pi_half)

    # Repeated CPMG blocks
    for _ in range(n_pulses):
        # Free evolution (modeled as very weak pulse)
        # For free evolution, we use omega=0 but that may cause issues
        # Instead, use a continuous pulse with omega~0 and delta=detuning
        seq.add_continuous_pulse(omega=1e-10, axis=[0, 0, 1], delta=delta, tau=interval)

        # π pulse on y-axis
        seq.add_continuous_pulse(omega=omega_pi, axis=[0, 1, 0], delta=0, tau=tau_pi)

        # Free evolution
        seq.add_continuous_pulse(omega=1e-10, axis=[0, 0, 1], delta=delta, tau=interval)

    # Final π/2 pulse on x-axis
    seq.add_continuous_pulse(omega=omega_pi, axis=[1, 0, 0], delta=0, tau=tau_pi_half)

    return seq


# =============================================================================
# Main Plotting
# =============================================================================

def main():
    # Set up matplotlib
    rcParams['font.size'] = FONT_SIZE
    rcParams['axes.linewidth'] = 1.2
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2

    # Frequency array
    if USE_LOG_SCALE:
        frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ_POINTS)
    else:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ_POINTS)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    print("Building and computing filter functions...")
    print(f"Total free evolution time: tau = {TAU_TOTAL}")
    print(f"pi-pulse Rabi frequency: Omega = {OMEGA_PI:.2f} (duration = {TAU_PI:.4f})")
    print()

    for n in N_PULSES_LIST:
        color = COLORS.get(n, '#333333')

        # Build instantaneous CPMG
        seq_inst = build_instantaneous_cpmg(TAU_TOTAL, n, DELTA)
        ff_inst = seq_inst.get_filter_function_calculator()
        susc_inst = ff_inst.noise_susceptibility(frequencies)

        # Build continuous CPMG
        seq_cont = build_continuous_cpmg(TAU_TOTAL, n, OMEGA_PI, DELTA)
        ff_cont = seq_cont.get_filter_function_calculator()
        susc_cont = ff_cont.noise_susceptibility(frequencies)

        print(f"N = {n}:")
        print(f"  Instantaneous duration: {seq_inst.total_duration():.4f}")
        print(f"  Continuous duration: {seq_cont.total_duration():.4f}")

        # Plot
        label_inst = f'N={n} (instant)'
        label_cont = f'N={n} (continuous)'

        if USE_LOG_SCALE:
            ax.loglog(frequencies, susc_inst,
                      linestyle=STYLE_INSTANTANEOUS, color=color,
                      linewidth=LINE_WIDTH, label=label_inst)
            ax.loglog(frequencies, susc_cont,
                      linestyle=STYLE_CONTINUOUS, color=color,
                      linewidth=LINE_WIDTH, label=label_cont, alpha=0.8)
        else:
            ax.plot(frequencies, susc_inst,
                    linestyle=STYLE_INSTANTANEOUS, color=color,
                    linewidth=LINE_WIDTH, label=label_inst)
            ax.plot(frequencies, susc_cont,
                    linestyle=STYLE_CONTINUOUS, color=color,
                    linewidth=LINE_WIDTH, label=label_cont, alpha=0.8)

    ax.set_xlabel(r'Angular frequency $\omega$')
    ax.set_ylabel(r'$|F(\omega)|^2$')
    ax.set_title(f'CPMG Filter Functions: Instantaneous vs Continuous Pulses\n'
                 f'($\\tau_{{total}} = {TAU_TOTAL}$, '
                 f'$\\Omega_\\pi = {OMEGA_PI:.1f}$)')
    ax.legend(loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    # Add annotation for line styles
    ax.annotate('Solid: Instantaneous\nDashed: Continuous',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    if SAVE_FIGURE:
        output_dir = Path(__file__).parent.parent / 'figures'
        output_dir.mkdir(exist_ok=True)
        output_path = Path(__file__).parent.parent / OUTPUT_PATH
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    main()
