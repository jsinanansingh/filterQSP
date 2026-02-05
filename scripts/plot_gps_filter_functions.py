"""
Plot Global Phase Spectroscopy filter functions.

This script generates plots comparing filter functions for the GPS protocol
with different numbers of Rabi cycles and different measurement types on
the metastable (clock) transition.

Usage:
    python scripts/plot_gps_filter_functions.py

Output:
    figures/gps_filter_functions.pdf
    figures/gps_measurement_comparison.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.global_phase_spectroscopy import (
    GlobalPhaseSpectroscopySequence,
    gps_filter_functions_comparison,
)


def plot_filter_functions_vs_cycles(system, omega, n_cycles_list, frequencies,
                                     measurement_type='z', ax=None):
    """
    Plot filter functions for different numbers of Rabi cycles.

    Parameters
    ----------
    system : ThreeLevelClock
        Three-level system
    omega : float
        Rabi frequency
    n_cycles_list : list of int
        Numbers of cycles to compare
    frequencies : np.ndarray
        Frequency array
    measurement_type : str
        Clock measurement type ('z', 'x', 'y')
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_cycles_list)))

    for i, n_cycles in enumerate(n_cycles_list):
        gps = GlobalPhaseSpectroscopySequence(system, n_cycles=n_cycles, omega=omega)
        ff = gps.get_filter_function_calculator()

        ff_values = ff.filter_function_for_measurement(frequencies, measurement_type)

        # Normalize by total time squared for comparison
        T_total = gps.total_time
        ff_normalized = ff_values / T_total**2

        ax.semilogy(frequencies / omega, ff_normalized + 1e-20,
                   color=colors[i], linewidth=1.5,
                   label=f'm = {n_cycles} cycles')

    # Mark Rabi frequency harmonics
    for n in range(1, 4):
        ax.axvline(n, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_xlabel(r'$\omega / \Omega$ (normalized frequency)', fontsize=12)
    ax.set_ylabel(r'$|F(\omega)|^2 / T^2$ (normalized)', fontsize=12)
    ax.set_title(f'GPS Filter Function: clock $\\sigma_{measurement_type}$ measurement',
                fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, frequencies[-1] / omega])
    ax.grid(True, alpha=0.3)

    return fig


def plot_measurement_comparison(system, omega, n_cycles, frequencies, ax=None):
    """
    Compare filter functions for different clock measurement types.

    Parameters
    ----------
    system : ThreeLevelClock
        Three-level system
    omega : float
        Rabi frequency
    n_cycles : int
        Number of Rabi cycles
    frequencies : np.ndarray
        Frequency array
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    gps = GlobalPhaseSpectroscopySequence(system, n_cycles=n_cycles, omega=omega)
    ff = gps.get_filter_function_calculator()
    T_total = gps.total_time

    measurement_types = ['z', 'x', 'y']
    labels = [r'$\sigma_z$ (population difference)',
              r'$\sigma_x$ (Re coherence)',
              r'$\sigma_y$ (Im coherence)']
    colors = ['C0', 'C1', 'C2']
    linestyles = ['-', '--', ':']

    for meas_type, label, color, ls in zip(measurement_types, labels, colors, linestyles):
        ff_values = ff.filter_function_for_measurement(frequencies, meas_type)
        ff_normalized = ff_values / T_total**2

        ax.semilogy(frequencies / omega, ff_normalized + 1e-20,
                   color=color, linestyle=ls, linewidth=2,
                   label=label)

    # Mark Rabi frequency harmonics
    for n in range(1, 4):
        ax.axvline(n, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_xlabel(r'$\omega / \Omega$ (normalized frequency)', fontsize=12)
    ax.set_ylabel(r'$|F(\omega)|^2 / T^2$ (normalized)', fontsize=12)
    ax.set_title(f'GPS Filter Functions: m = {n_cycles} cycles, different measurements',
                fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, frequencies[-1] / omega])
    ax.grid(True, alpha=0.3)

    return fig


def plot_sensitivity_heatmap(system, omega, n_cycles_array, frequencies,
                              measurement_type='z', ax=None):
    """
    Create heatmap of filter function vs frequency and number of cycles.

    Parameters
    ----------
    system : ThreeLevelClock
        Three-level system
    omega : float
        Rabi frequency
    n_cycles_array : np.ndarray
        Array of cycle counts
    frequencies : np.ndarray
        Frequency array
    measurement_type : str
        Clock measurement type
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Build 2D array of filter functions
    ff_matrix = np.zeros((len(n_cycles_array), len(frequencies)))

    for i, n_cycles in enumerate(n_cycles_array):
        gps = GlobalPhaseSpectroscopySequence(system, n_cycles=int(n_cycles), omega=omega)
        ff = gps.get_filter_function_calculator()
        T_total = gps.total_time

        ff_values = ff.filter_function_for_measurement(frequencies, measurement_type)
        ff_matrix[i, :] = ff_values / T_total**2

    # Plot heatmap
    extent = [frequencies[0] / omega, frequencies[-1] / omega,
              n_cycles_array[0], n_cycles_array[-1]]

    im = ax.imshow(np.log10(ff_matrix + 1e-20), aspect='auto', origin='lower',
                   extent=extent, cmap='viridis')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}(|F|^2 / T^2)$', fontsize=11)

    # Mark Rabi frequency harmonics
    for n in range(1, 5):
        ax.axvline(n, color='white', linestyle='--', alpha=0.5, linewidth=0.5)

    ax.set_xlabel(r'$\omega / \Omega$ (normalized frequency)', fontsize=12)
    ax.set_ylabel('Number of Rabi cycles m', fontsize=12)
    ax.set_title(f'GPS Sensitivity: clock $\\sigma_{measurement_type}$ measurement',
                fontsize=12)

    return fig


def main():
    """Generate all GPS filter function plots."""
    # Create output directory
    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)

    # Setup system
    system = ThreeLevelClock()
    omega = 2 * np.pi  # 1 Hz Rabi frequency (normalized units)

    # Frequency array (in units of Rabi frequency)
    frequencies = np.linspace(0.01, 5 * omega, 500)

    print("Generating GPS filter function plots...")

    # =========================================================================
    # Plot 1: Filter functions vs number of cycles
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

    n_cycles_list = [1, 3, 5, 10, 20]

    for i, meas_type in enumerate(['z', 'x', 'y']):
        plot_filter_functions_vs_cycles(
            system, omega, n_cycles_list, frequencies,
            measurement_type=meas_type, ax=axes1[i]
        )
        axes1[i].set_title(f'clock $\\sigma_{meas_type}$ measurement')

    plt.tight_layout()
    fig1.savefig(fig_dir / 'gps_filter_functions.pdf', dpi=150, bbox_inches='tight')
    fig1.savefig(fig_dir / 'gps_filter_functions.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'gps_filter_functions.pdf'}")

    # =========================================================================
    # Plot 2: Measurement type comparison for fixed cycles
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    for i, n_cycles in enumerate([3, 10, 30]):
        plot_measurement_comparison(
            system, omega, n_cycles, frequencies, ax=axes2[i]
        )
        axes2[i].set_title(f'm = {n_cycles} Rabi cycles')

    plt.tight_layout()
    fig2.savefig(fig_dir / 'gps_measurement_comparison.pdf', dpi=150, bbox_inches='tight')
    fig2.savefig(fig_dir / 'gps_measurement_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'gps_measurement_comparison.pdf'}")

    # =========================================================================
    # Plot 3: Sensitivity heatmap
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))

    n_cycles_array = np.arange(1, 31)
    frequencies_heatmap = np.linspace(0.01, 4 * omega, 200)

    for i, meas_type in enumerate(['z', 'x', 'y']):
        plot_sensitivity_heatmap(
            system, omega, n_cycles_array, frequencies_heatmap,
            measurement_type=meas_type, ax=axes3[i]
        )

    plt.tight_layout()
    fig3.savefig(fig_dir / 'gps_sensitivity_heatmap.pdf', dpi=150, bbox_inches='tight')
    fig3.savefig(fig_dir / 'gps_sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'gps_sensitivity_heatmap.pdf'}")

    # =========================================================================
    # Plot 4: Comparison with standard Ramsey
    # =========================================================================
    from quantum_pulse_suite.core.multilevel import multilevel_ramsey

    fig4, ax4 = plt.subplots(figsize=(8, 5))

    # GPS with 10 cycles
    gps = GlobalPhaseSpectroscopySequence(system, n_cycles=10, omega=omega)
    ff_gps = gps.get_filter_function_calculator()
    T_gps = gps.total_time

    # Ramsey with same total time
    ramsey = multilevel_ramsey(system, system.probe, tau=T_gps)
    ff_ramsey = ramsey.get_filter_function_calculator()

    # Get filter functions
    _, _, Fz_gps = ff_gps.filter_function(frequencies)
    _, _, Fz_ramsey = ff_ramsey.filter_function(frequencies)

    # Normalize
    Fz_gps_norm = Fz_gps**2 / T_gps**2
    Fz_ramsey_norm = Fz_ramsey**2 / T_gps**2

    ax4.semilogy(frequencies / omega, Fz_gps_norm + 1e-20,
                'C0-', linewidth=2, label=f'GPS (m={10} cycles)')
    ax4.semilogy(frequencies / omega, Fz_ramsey_norm + 1e-20,
                'C1--', linewidth=2, label=f'Ramsey (T={T_gps:.1f})')

    ax4.set_xlabel(r'$\omega / \Omega$ (normalized frequency)', fontsize=12)
    ax4.set_ylabel(r'$|F_z(\omega)|^2 / T^2$', fontsize=12)
    ax4.set_title('GPS vs Ramsey: Dephasing Sensitivity Comparison', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.set_xlim([0, 4])
    ax4.grid(True, alpha=0.3)

    fig4.savefig(fig_dir / 'gps_vs_ramsey.pdf', dpi=150, bbox_inches='tight')
    fig4.savefig(fig_dir / 'gps_vs_ramsey.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'gps_vs_ramsey.pdf'}")

    plt.close('all')
    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
