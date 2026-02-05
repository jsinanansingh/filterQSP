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


def plot_filter_functions_vs_cycles(system, T_total, n_cycles_list, frequencies,
                                     measurement_type='z', ax=None):
    """
    Plot filter functions for different numbers of Rabi cycles at fixed total time.

    Parameters
    ----------
    system : ThreeLevelClock
        Three-level system
    T_total : float
        Fixed total protocol time
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
        # Adjust omega so that n_cycles fit in the fixed total time
        # T_total = n_cycles * (2*pi / omega) => omega = 2*pi * n_cycles / T_total
        omega = 2 * np.pi * n_cycles / T_total
        gps = GlobalPhaseSpectroscopySequence(system, n_cycles=n_cycles, omega=omega)
        ff = gps.get_filter_function_calculator()

        ff_values = ff.filter_function_for_measurement(frequencies, measurement_type)

        # Normalize by total time squared for comparison
        ff_normalized = ff_values / T_total**2

        ax.semilogy(frequencies * T_total / (2 * np.pi), ff_normalized + 1e-20,
                   color=colors[i], linewidth=1.5,
                   label=f'm = {n_cycles} cycles')

    # Mark cycle count positions (peaks should appear near these)
    for n in n_cycles_list:
        ax.axvline(n, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_xlabel(r'$\omega T / 2\pi$ (cycles)', fontsize=12)
    ax.set_ylabel(r'$|F(\omega)|^2 / T^2$ (normalized)', fontsize=12)
    ax.set_title(f'GPS Filter Function: clock $\\sigma_{measurement_type}$ measurement',
                fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, frequencies[-1] * T_total / (2 * np.pi)])
    ax.grid(True, alpha=0.3)

    return fig


def plot_measurement_comparison(system, T_total, n_cycles, frequencies, ax=None):
    """
    Compare filter functions for different clock measurement types.

    Parameters
    ----------
    system : ThreeLevelClock
        Three-level system
    T_total : float
        Fixed total protocol time
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

    # Adjust omega so that n_cycles fit in the fixed total time
    omega = 2 * np.pi * n_cycles / T_total
    gps = GlobalPhaseSpectroscopySequence(system, n_cycles=n_cycles, omega=omega)
    ff = gps.get_filter_function_calculator()

    measurement_types = ['z', 'x', 'y']
    labels = [r'$\sigma_z$ (population difference)',
              r'$\sigma_x$ (Re coherence)',
              r'$\sigma_y$ (Im coherence)']
    colors = ['C0', 'C1', 'C2']
    linestyles = ['-', '--', ':']

    for meas_type, label, color, ls in zip(measurement_types, labels, colors, linestyles):
        ff_values = ff.filter_function_for_measurement(frequencies, meas_type)
        ff_normalized = ff_values / T_total**2

        ax.semilogy(frequencies * T_total / (2 * np.pi), ff_normalized + 1e-20,
                   color=color, linestyle=ls, linewidth=2,
                   label=label)

    # Mark the cycle count position (peak should appear near here)
    ax.axvline(n_cycles, color='gray', linestyle='--', alpha=0.5, linewidth=1,
               label=f'm = {n_cycles}')

    ax.set_xlabel(r'$\omega T / 2\pi$ (cycles)', fontsize=12)
    ax.set_ylabel(r'$|F(\omega)|^2 / T^2$ (normalized)', fontsize=12)
    ax.set_title(f'GPS Filter Functions: m = {n_cycles} cycles, different measurements',
                fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, frequencies[-1] * T_total / (2 * np.pi)])
    ax.grid(True, alpha=0.3)

    return fig


def plot_sensitivity_heatmap(system, T_total, n_cycles_array, frequencies,
                              measurement_type='z', ax=None):
    """
    Create heatmap of filter function vs frequency and number of cycles.

    Parameters
    ----------
    system : ThreeLevelClock
        Three-level system
    T_total : float
        Fixed total protocol time
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
        # Adjust omega so that n_cycles fit in the fixed total time
        omega = 2 * np.pi * n_cycles / T_total
        gps = GlobalPhaseSpectroscopySequence(system, n_cycles=int(n_cycles), omega=omega)
        ff = gps.get_filter_function_calculator()

        ff_values = ff.filter_function_for_measurement(frequencies, measurement_type)
        ff_matrix[i, :] = ff_values / T_total**2

    # Plot heatmap with x-axis in cycles (omega * T / 2pi)
    freq_cycles = frequencies * T_total / (2 * np.pi)
    extent = [freq_cycles[0], freq_cycles[-1],
              n_cycles_array[0], n_cycles_array[-1]]

    im = ax.imshow(np.log10(ff_matrix + 1e-20), aspect='auto', origin='lower',
                   extent=extent, cmap='viridis')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}(|F|^2 / T^2)$', fontsize=11)

    # Mark diagonal where frequency matches cycle count (peak positions)
    ax.plot([n_cycles_array[0], n_cycles_array[-1]],
            [n_cycles_array[0], n_cycles_array[-1]],
            'w--', alpha=0.7, linewidth=1.5, label='Peak location')

    ax.set_xlabel(r'$\omega T / 2\pi$ (cycles)', fontsize=12)
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

    # Fixed total protocol time (all protocols will have the same duration)
    T_total = 2 * np.pi  # normalized units

    # Frequency array - range chosen so that we can see peaks up to ~25 cycles
    # x-axis will be omega * T / (2*pi), so max_freq * T / (2*pi) = max_cycles_shown
    max_cycles_shown = 25
    frequencies = np.linspace(0.01, max_cycles_shown * 2 * np.pi / T_total, 500)

    print("Generating GPS filter function plots...")
    print(f"  Fixed total time T = {T_total}")

    # =========================================================================
    # Plot 1: Filter functions vs number of cycles
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

    n_cycles_list = [1, 3, 5, 10, 20]

    for i, meas_type in enumerate(['z', 'x', 'y']):
        plot_filter_functions_vs_cycles(
            system, T_total, n_cycles_list, frequencies,
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
            system, T_total, n_cycles, frequencies, ax=axes2[i]
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
    # Frequency range for heatmap - show up to 35 cycles
    max_cycles_heatmap = 35
    frequencies_heatmap = np.linspace(0.01, max_cycles_heatmap * 2 * np.pi / T_total, 200)

    for i, meas_type in enumerate(['z', 'x', 'y']):
        plot_sensitivity_heatmap(
            system, T_total, n_cycles_array, frequencies_heatmap,
            measurement_type=meas_type, ax=axes3[i]
        )

    plt.tight_layout()
    fig3.savefig(fig_dir / 'gps_sensitivity_heatmap.pdf', dpi=150, bbox_inches='tight')
    fig3.savefig(fig_dir / 'gps_sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'gps_sensitivity_heatmap.pdf'}")

    # =========================================================================
    # Plot 4: Comparison of Ramsey, Rabi, and GPS protocols (σ_y measurement)
    # =========================================================================
    from quantum_pulse_suite.core.multilevel import multilevel_ramsey, MultiLevelPulseSequence

    fig4, ax4 = plt.subplots(figsize=(10, 6))

    # Frequency array for comparison plot - show up to 12 cycles
    freq_compare = np.linspace(0.01, 12 * 2 * np.pi / T_total, 500)
    freq_cycles = freq_compare * T_total / (2 * np.pi)

    # --- Ramsey with optimal detuning for sensitivity ---
    # Optimal detuning puts the working point at max slope: delta*T = pi/2
    delta_ramsey_opt = np.pi / (2 * T_total)
    ramsey = multilevel_ramsey(system, system.probe, tau=T_total, delta=delta_ramsey_opt)
    ff_ramsey = ramsey.get_filter_function_calculator()
    _, Fy_ramsey, _ = ff_ramsey.filter_function(freq_compare)
    Fy_ramsey_norm = Fy_ramsey**2 / T_total**2

    ax4.semilogy(freq_cycles, Fy_ramsey_norm + 1e-20,
                'C0-', linewidth=2, label='Ramsey (optimal detuning)')

    # --- Rabi with optimal detuning for sensitivity ---
    # For Rabi, create a continuous drive on the probe transition
    # Optimal detuning for Rabi: delta = Omega / sqrt(3) gives max slope
    # We want to fill total time T with some Rabi oscillations
    # Let's use 1 Rabi cycle worth of omega, then add detuning
    omega_rabi = 2 * np.pi / T_total  # 1 cycle in time T
    delta_rabi_opt = omega_rabi / np.sqrt(3)

    # Create Rabi sequence using multilevel pulse sequence
    rabi_seq = MultiLevelPulseSequence(system, system.probe)
    rabi_seq.add_continuous_pulse(omega_rabi, axis=[1, 0, 0], tau=T_total, delta=delta_rabi_opt)
    ff_rabi = rabi_seq.get_filter_function_calculator()
    _, Fy_rabi, _ = ff_rabi.filter_function(freq_compare)
    Fy_rabi_norm = Fy_rabi**2 / T_total**2

    ax4.semilogy(freq_cycles, Fy_rabi_norm + 1e-20,
                'C1--', linewidth=2, label='Rabi (optimal detuning)')

    # --- GPS with m = 1, 4, 8 cycles ---
    gps_cycles = [1, 4, 8]
    gps_colors = ['C2', 'C3', 'C4']
    gps_linestyles = [':', '-.', '-']

    for n_cyc, color, ls in zip(gps_cycles, gps_colors, gps_linestyles):
        omega_gps = 2 * np.pi * n_cyc / T_total
        gps = GlobalPhaseSpectroscopySequence(system, n_cycles=n_cyc, omega=omega_gps)
        ff_gps = gps.get_filter_function_calculator()
        # Use σ_y measurement for GPS
        Fy_gps_sq = ff_gps.filter_function_for_measurement(freq_compare, 'y')
        Fy_gps_norm = Fy_gps_sq / T_total**2

        ax4.semilogy(freq_cycles, Fy_gps_norm + 1e-20,
                    color=color, linestyle=ls, linewidth=2,
                    label=f'GPS (m={n_cyc})')

    ax4.set_xlabel(r'$\omega T / 2\pi$ (cycles)', fontsize=12)
    ax4.set_ylabel(r'$|F_y(\omega)|^2 / T^2$', fontsize=12)
    ax4.set_title(r'Protocol Comparison: $\sigma_y$ measurement (same total time T)', fontsize=12)
    ax4.legend(fontsize=10, loc='upper right')
    ax4.set_xlim([0, 12])
    ax4.grid(True, alpha=0.3)

    fig4.savefig(fig_dir / 'gps_vs_ramsey.pdf', dpi=150, bbox_inches='tight')
    fig4.savefig(fig_dir / 'gps_vs_ramsey.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'gps_vs_ramsey.pdf'}")

    # =========================================================================
    # Plot 5: Qubit (2-level) comparison - σ_y projective measurement
    # =========================================================================
    from quantum_pulse_suite.core.pulse_sequence import (
        ramsey_sequence, continuous_rabi_sequence, ContinuousPulseSequence
    )

    fig5, ax5 = plt.subplots(figsize=(10, 6))

    # Frequency array for comparison plot - logarithmic from 0.1 to 100
    freq_compare = np.logspace(-1, 2, 500)  # 0.1 to 100 rad/s

    def get_final_rotation_matrix(f, g):
        """
        Compute the SO(3) rotation matrix from SU(2) parameters f, g.

        For U = [[f, ig], [ig*, f*]], the rotation matrix R satisfies:
        U† σ_j U = sum_k R_jk σ_k
        """
        # Components for rotation matrix
        # Using standard SU(2) to SO(3) map
        a, b = np.real(f), np.imag(f)
        c, d = np.real(g), np.imag(g)

        R = np.array([
            [a**2 + c**2 - b**2 - d**2, 2*(c*d - a*b), 2*(a*d + b*c)],
            [2*(a*b + c*d), a**2 + d**2 - b**2 - c**2, 2*(b*d - a*c)],
            [2*(b*c - a*d), 2*(a*c + b*d), a**2 + b**2 - c**2 - d**2]
        ])
        return R

    def compute_kubo_sensitivity(seq, frequencies, B_lab=np.array([0, 1, 0])):
        """
        Compute Kubo formula sensitivity: |F|^2 - (B·F)^2

        B_lab is the measurement Bloch vector in lab frame (default σ_y).
        B in interaction frame = R^T · B_lab where R is the sequence rotation.
        """
        # Get filter function
        ff = seq.get_filter_function_calculator()
        Fx, Fy, Fz = ff.filter_function(frequencies)

        # Get final f, g from the sequence
        if hasattr(seq, '_polynomial_segments') and seq._polynomial_segments:
            F_func, G_func, _, end_time = seq._polynomial_segments[-1]
            f_final = F_func(end_time)
            g_final = G_func(end_time)
        else:
            # Default to identity if no segments
            f_final, g_final = 1.0, 0.0

        # Compute rotation matrix and transform B to interaction frame
        R = get_final_rotation_matrix(f_final, g_final)
        B = R.T @ B_lab  # B in interaction frame

        # Compute |F|^2 - (B·F)^2
        F_mag_sq = Fx**2 + Fy**2 + Fz**2
        B_dot_F = B[0]*Fx + B[1]*Fy + B[2]*Fz
        sensitivity = F_mag_sq - B_dot_F**2

        return sensitivity, B

    # --- Ramsey with instantaneous pulses ---
    delta_ramsey = 0.25  # detuning 1/4
    ramsey_instant = ramsey_sequence(tau=T_total, delta=delta_ramsey)
    ramsey_instant.compute_polynomials()  # Ensure polynomials are computed

    sens_ramsey_instant, B_ramsey = compute_kubo_sensitivity(ramsey_instant, freq_compare)
    F_ramsey_instant_norm = sens_ramsey_instant / (T_total**2 * freq_compare**2)

    ax5.loglog(freq_compare, F_ramsey_instant_norm + 1e-20,
              'C0-', linewidth=2, label=f'Ramsey (instant)')

    # --- Ramsey with continuous pulses ---
    t_pi2 = 1e-2  # Duration of π/2 pulse
    omega_pulse = np.pi / (2 * t_pi2)  # Rabi frequency for π/2 pulses
    tau_free = T_total - 2 * t_pi2  # Free evolution time

    ramsey_cont = ContinuousPulseSequence()
    ramsey_cont.add_continuous_pulse(omega_pulse, [1, 0, 0], 0.0, t_pi2)        # π/2 pulse
    ramsey_cont.add_continuous_pulse(0.0, [0, 0, 1], delta_ramsey, tau_free)    # Free evolution with detuning
    ramsey_cont.add_continuous_pulse(omega_pulse, [1, 0, 0], 0.0, t_pi2)        # π/2 pulse
    ramsey_cont.compute_polynomials()

    sens_ramsey_cont, _ = compute_kubo_sensitivity(ramsey_cont, freq_compare)
    F_ramsey_cont_norm = sens_ramsey_cont / (T_total**2 * freq_compare**2)

    ax5.loglog(freq_compare, F_ramsey_cont_norm + 1e-20,
              'C0--', linewidth=2, label='Ramsey (continuous)')

    # --- Continuous Rabi with m = 1, 4, 8 cycles ---
    rabi_cycles = [1, 4, 8]
    rabi_colors = ['C1', 'C2', 'C3']
    rabi_linestyles = ['-', ':', '-.']

    for n_cyc, color, ls in zip(rabi_cycles, rabi_colors, rabi_linestyles):
        omega_rabi_n = 2 * np.pi * n_cyc / T_total
        rabi_n = continuous_rabi_sequence(omega=omega_rabi_n, tau=T_total, delta=0.0)
        rabi_n.compute_polynomials()

        sens_rabi_n, B_rabi = compute_kubo_sensitivity(rabi_n, freq_compare)
        F_rabi_n_norm = sens_rabi_n / (T_total**2 * freq_compare**2)

        ax5.loglog(freq_compare, F_rabi_n_norm + 1e-20,
                  color=color, linestyle=ls, linewidth=2,
                  label=f'Rabi m={n_cyc}')

    # Add power law reference lines
    freq_ref = np.logspace(-1, 2, 100)  # 0.1 to 100 rad/s

    # Reference amplitude and frequency (adjust to fit in plot)
    ref_amp = 0.01
    ref_freq = 1.0  # Reference frequency for normalization

    # ω^{-2} (Ramsey-like falloff)
    power_law_2 = ref_amp * (freq_ref / ref_freq)**(-2)
    ax5.loglog(freq_ref, power_law_2, 'k-', linewidth=1.5, alpha=0.5, label=r'$\omega^{-2}$')

    # ω^{-4}
    power_law_4 = ref_amp * (freq_ref / ref_freq)**(-4)
    ax5.loglog(freq_ref, power_law_4, 'k--', linewidth=1.5, alpha=0.5, label=r'$\omega^{-4}$')

    # ω^{-6}
    power_law_6 = ref_amp * (freq_ref / ref_freq)**(-6)
    ax5.loglog(freq_ref, power_law_6, 'k:', linewidth=1.5, alpha=0.5, label=r'$\omega^{-6}$')

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax5.set_ylabel(r'$(|\mathbf{F}|^2 - (\mathbf{B}\cdot\mathbf{F})^2) / (\omega^2 T^2)$', fontsize=12)
    ax5.set_title(r'Qubit: $\sigma_y$ measurement sensitivity (Kubo formula)', fontsize=12)
    ax5.legend(fontsize=9, loc='lower left', ncol=2)
    ax5.set_xlim([0.1, 100])
    ax5.set_ylim([1e-16, 10])
    ax5.grid(True, alpha=0.3, which='both')

    fig5.savefig(fig_dir / 'qubit_protocol_comparison.pdf', dpi=150, bbox_inches='tight')
    fig5.savefig(fig_dir / 'qubit_protocol_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_dir / 'qubit_protocol_comparison.pdf'}")

    plt.close('all')
    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
