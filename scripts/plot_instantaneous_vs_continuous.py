"""
Plot comparing filter functions for analogous instantaneous vs continuous pulse sequences.

This script compares:
- Continuous: rotation about y-axis by angle θ with Rabi frequency Ω and duration τ
- Instantaneous: R_x(π) → free evolution (R_z) → R_x(-π)

These sequences achieve analogous QSP polynomials via conjugation:
    R_x(π) · R_z(θ) · R_x(-π) = σ_x · R_z(θ) · σ_x = R_z(-θ)

The continuous R_y(θ) and the conjugated instantaneous sequence both implement
rotations, but their filter functions differ due to the different time structure.

Usage:
    python scripts/plot_instantaneous_vs_continuous.py
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
    SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY,
)

# =============================================================================
# Plot Configuration - EDIT THESE PARAMETERS
# =============================================================================

# Pulse parameters
THETA = np.pi / 2        # Rotation angle
DELTA = 0.5              # Detuning (common to both)
OMEGA_CONTINUOUS = 2.0   # Rabi frequency for continuous pulse
TAU_CONTINUOUS = THETA / np.sqrt(OMEGA_CONTINUOUS**2 + DELTA**2)  # Duration to achieve θ

# For instantaneous sequence: free evolution time to get R_z(θ)
TAU_FREE = THETA / DELTA if DELTA != 0 else 1.0

# Frequency range for filter function
FREQ_MIN = 0.1
FREQ_MAX = 50
N_FREQ_POINTS = 500

# Plot styling
FIGURE_SIZE = (8, 6)
DPI = 150
FONT_SIZE = 12
LINE_WIDTH = 2.0
USE_LOG_SCALE = True

# Colors
COLOR_CONTINUOUS = '#1f77b4'  # Blue
COLOR_INSTANTANEOUS = '#d62728'  # Red

# Output
SAVE_FIGURE = True
OUTPUT_PATH = 'figures/instantaneous_vs_continuous_filter.pdf'

# =============================================================================
# Build Sequences
# =============================================================================

def build_continuous_y_rotation(omega, delta, tau):
    """
    Build a continuous pulse sequence: rotation about y-axis.

    H = (Ω/2) σ_y + (δ/2) σ_z
    """
    seq = ContinuousPulseSequence()
    seq.add_continuous_pulse(omega=omega, axis=[0, 1, 0], delta=delta, tau=tau)
    return seq


def build_instantaneous_conjugated(theta, delta):
    """
    Build instantaneous sequence: R_x(π) · free_evolution · R_x(-π)

    The free evolution implements R_z(δτ) where τ = θ/δ.
    The full sequence implements a conjugated z-rotation.
    """
    tau_free = theta / delta if abs(delta) > 1e-10 else 1.0

    seq = InstantaneousPulseSequence()
    seq.add_instant_pulse([1, 0, 0], np.pi)      # R_x(π)
    seq.add_free_evolution(tau_free, delta)      # R_z(θ) via free evolution
    seq.add_instant_pulse([1, 0, 0], -np.pi)     # R_x(-π)
    return seq


def verify_conjugation():
    """
    Verify that the conjugation relation holds:
    R_x(π) · R_z(θ) · R_x(-π) should equal R_z(-θ) (up to global phase)
    """
    from scipy.linalg import expm

    theta = np.pi / 4

    # R_x(π) = exp(-i σ_x π/2) = -i σ_x
    R_x_pi = expm(-1j * SIGMA_X * np.pi / 2)
    R_x_minus_pi = expm(1j * SIGMA_X * np.pi / 2)
    R_z_theta = expm(-1j * SIGMA_Z * theta / 2)

    # Conjugated sequence
    conjugated = R_x_pi @ R_z_theta @ R_x_minus_pi

    # Expected: R_z(-θ)
    R_z_minus_theta = expm(1j * SIGMA_Z * theta / 2)

    print("Verification of conjugation relation:")
    print(f"R_x(pi) . R_z(theta) . R_x(-pi) =\n{conjugated}")
    print(f"\nR_z(-theta) =\n{R_z_minus_theta}")
    print(f"\nDifference norm: {np.linalg.norm(conjugated - R_z_minus_theta):.2e}")

    # Check if equal up to global phase
    if np.linalg.norm(conjugated - R_z_minus_theta) < 1e-10:
        print("[OK] Conjugation verified: R_x(pi) . R_z(theta) . R_x(-pi) = R_z(-theta)")
    else:
        # Check for global phase
        ratio = conjugated[0, 0] / R_z_minus_theta[0, 0] if abs(R_z_minus_theta[0, 0]) > 1e-10 else 1
        scaled = conjugated / ratio
        if np.linalg.norm(scaled - R_z_minus_theta) < 1e-10:
            print(f"[OK] Equal up to global phase: {ratio:.4f}")
        else:
            print("[WARN] Conjugation relation does not hold as expected")

    return conjugated, R_z_minus_theta


# =============================================================================
# Main Plotting
# =============================================================================

def main():
    # Set up matplotlib
    rcParams['font.size'] = FONT_SIZE
    rcParams['axes.linewidth'] = 1.2
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2

    # Verify conjugation
    print("=" * 60)
    verify_conjugation()
    print("=" * 60)

    # Build sequences
    seq_continuous = build_continuous_y_rotation(OMEGA_CONTINUOUS, DELTA, TAU_CONTINUOUS)
    seq_instantaneous = build_instantaneous_conjugated(THETA, DELTA)

    print(f"\nSequence parameters:")
    print(f"  Continuous: Omega={OMEGA_CONTINUOUS}, delta={DELTA}, tau={TAU_CONTINUOUS:.4f}")
    print(f"  Continuous total duration: {seq_continuous.total_duration():.4f}")
    print(f"  Instantaneous total duration: {seq_instantaneous.total_duration():.4f}")

    # Get filter functions
    ff_continuous = seq_continuous.get_filter_function_calculator()
    ff_instantaneous = seq_instantaneous.get_filter_function_calculator()

    # Frequency array
    if USE_LOG_SCALE:
        frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ_POINTS)
    else:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ_POINTS)

    # Compute filter functions
    susc_continuous = ff_continuous.noise_susceptibility(frequencies)
    susc_instantaneous = ff_instantaneous.noise_susceptibility(frequencies)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if USE_LOG_SCALE:
        ax.loglog(frequencies, susc_continuous,
                  color=COLOR_CONTINUOUS, linewidth=LINE_WIDTH,
                  label=f'Continuous $R_y(\\theta)$, $\\Omega={OMEGA_CONTINUOUS}$')
        ax.loglog(frequencies, susc_instantaneous,
                  color=COLOR_INSTANTANEOUS, linewidth=LINE_WIDTH,
                  label=f'Instantaneous $R_x(\\pi) R_z R_x(-\\pi)$')
    else:
        ax.plot(frequencies, susc_continuous,
                color=COLOR_CONTINUOUS, linewidth=LINE_WIDTH,
                label=f'Continuous $R_y(\\theta)$, $\\Omega={OMEGA_CONTINUOUS}$')
        ax.plot(frequencies, susc_instantaneous,
                color=COLOR_INSTANTANEOUS, linewidth=LINE_WIDTH,
                label=f'Instantaneous $R_x(\\pi) R_z R_x(-\\pi)$')

    ax.set_xlabel(r'Angular frequency $\omega$')
    ax.set_ylabel(r'$|F(\omega)|^2$')
    ax.set_title(f'Filter Functions: Analogous Sequences\n'
                 f'($\\theta = \\pi/{int(np.pi/THETA)}$, $\\delta = {DELTA}$)')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

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
