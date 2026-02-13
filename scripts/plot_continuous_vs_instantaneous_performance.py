"""
Scatter: <delta sigma_y^2> for instantaneous vs continuous CPMG-4.

Uses the Kubo formula:
    <dB^2>(T) = (2/pi) int dw S(w) (|R(w)|^2 - |B(T).R(w)|^2)

where R(w) is the filter function Bloch vector (already includes 1/w),
and B(T) is the observable sigma_y in the toggling frame at time T.

Fixed total wall-clock time T. Each point = (noise PSD, amplitude A).
Four panels sweep the Rabi frequency Omega, showing the transition from
"instantaneous competitive" (low Omega, large pulse overhead) to
"continuous dominates" (high Omega, near-instantaneous limit).

Usage:
    python scripts/plot_continuous_vs_instantaneous_performance.py

Output:
    figures/continuous_vs_instantaneous_performance.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.integrate import simpson

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite import (
    ramsey_sequence,
    cpmg_sequence,
    continuous_cpmg_sequence,
    ContinuousPulseSequence,
    ColoredNoisePSD,
)

# =============================================================================
# Configuration
# =============================================================================

T_WALL = 2 * np.pi          # Fixed total wall-clock time
DELTA = 0.25                 # Detuning
N_CPMG = 4                   # Number of CPMG refocusing pulses

# Rabi frequencies for each panel (low -> high)
OMEGAS = [np.pi, 2*np.pi, 5*np.pi, 20*np.pi]

FREQ_MIN = 0.01
FREQ_MAX = 200
N_FREQ = 2000

# Observable: sigma_y
B_LAB = np.array([0, 1, 0])

# Noise amplitude sweep
AMPLITUDES = np.logspace(-2, 2, 25)

# Plot
FIGURE_SIZE = (14, 11)
DPI = 150
FONT_SIZE = 11
SAVE_FIGURE = True
OUTPUT_DIR = 'figures'

# =============================================================================
# Helper functions
# =============================================================================


def get_final_rotation_matrix(f, g):
    a, b = np.real(f), np.imag(f)
    c, d = np.real(g), np.imag(g)
    R = np.array([
        [a**2 + c**2 - b**2 - d**2, 2*(c*d - a*b), 2*(a*d + b*c)],
        [2*(a*b + c*d), a**2 + d**2 - b**2 - c**2, 2*(b*d - a*c)],
        [2*(b*c - a*d), 2*(a*c + b*d), a**2 + b**2 - c**2 - d**2]
    ])
    return R


def compute_kubo_sensitivity(seq, frequencies, B_lab=B_LAB):
    """Compute |R(w)|^2 - |B.R(w)|^2 for the Kubo formula."""
    ff = seq.get_filter_function_calculator()
    Fx, Fy, Fz = ff.filter_function(frequencies)

    U = seq.total_unitary()
    f_final = U[0, 0]
    g_final = U[0, 1] / 1j

    R = get_final_rotation_matrix(f_final, g_final)
    B = R.T @ B_lab

    F_mag_sq = np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2
    B_dot_F = B[0]*Fx + B[1]*Fy + B[2]*Fz
    sensitivity = F_mag_sq - np.abs(B_dot_F)**2
    return np.maximum(sensitivity, 0.0)


def compute_delta_sy_sq(seq, frequencies, S_beta_func):
    """
    Compute <delta sigma_y^2> via Kubo formula.

    <dB^2> = (2/pi) int dw S(w) (|R(w)|^2 - |B.R(w)|^2)

    The filter function R(w) already includes the 1/w factor
    (F(w) = i*exp(iwt0)*(exp(-iwt)-1)/w), so NO extra 1/w^2 division.
    """
    sensitivity = compute_kubo_sensitivity(seq, frequencies)
    S_vals = S_beta_func(frequencies)
    integrand = S_vals * sensitivity
    return 2 * simpson(integrand, x=frequencies) / np.pi


# =============================================================================
# Noise PSD shapes
# =============================================================================


def build_noise_shapes():
    """Build a diverse set of noise spectral densities at unit amplitude."""
    shapes = []

    # Power-law: S(w) = 1/|w|^alpha
    for alpha, label in [(0, 'White'), (1, '$1/f$'), (2, '$1/f^2$')]:
        S = ColoredNoisePSD.generic_power_law(
            amplitude=1.0, exponent=alpha, cutoff=1e-10
        )
        shapes.append((label, S, 'o', 'Power-law'))

    # Lorentzian: S(w) = gamma / (w^2 + gamma^2)
    for gamma in [0.1, 1.0, 10.0]:
        S = ColoredNoisePSD.lorentzian(amplitude=1.0, gamma=gamma)
        shapes.append((f'Lor $\\gamma\\!={gamma}$', S, 's', 'Lorentzian'))

    # Peaked Gaussian noise at many frequencies
    # Covers: low-freq, CPMG resonances, near Rabi frequencies, high-freq
    peak_freqs = [0.3, 0.7, 1.0, 2.0, 3.0, 5.0, 7.0,
                  10.0, 15.0, 20.0, 30.0, 50.0]
    for omega_0 in peak_freqs:
        sigma_rel = 0.3
        def make_peaked(w0, sig):
            def psd(w):
                w = np.asarray(w)
                return np.exp(-((w - w0)**2) / (2 * sig**2 * w0**2))
            return psd
        shapes.append((f'$\\omega_0\\!={omega_0}$',
                        make_peaked(omega_0, sigma_rel), 'D', 'Peaked'))

    return shapes


# =============================================================================
# Protocol builders
# =============================================================================


def build_cpmg_pair(omega_rabi):
    """
    Build instantaneous and continuous CPMG-N with FIXED wall-clock time T_WALL.

    Instantaneous: tau_free = T_WALL (delta-function pulses)
    Continuous:    tau_free = T_WALL - (N+1)*pi/Omega  (finite-duration pulses)
    """
    tau_pi2 = np.pi / (2 * omega_rabi)
    tau_pi = np.pi / omega_rabi
    pulse_overhead = 2 * tau_pi2 + N_CPMG * tau_pi
    tau_free_cont = T_WALL - pulse_overhead
    tau_free_inst = T_WALL

    if tau_free_cont <= 0:
        return None, None, tau_free_inst, -1

    seq_inst = cpmg_sequence(tau=tau_free_inst, n_pulses=N_CPMG, delta=DELTA)
    seq_inst.compute_polynomials()

    seq_cont = continuous_cpmg_sequence(
        omega=omega_rabi, tau=tau_free_cont, n_pulses=N_CPMG, delta=DELTA
    )
    seq_cont.compute_polynomials()

    return seq_inst, seq_cont, tau_free_inst, tau_free_cont


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

    fig_dir = Path(__file__).parent.parent / OUTPUT_DIR
    fig_dir.mkdir(exist_ok=True)
    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)

    shapes = build_noise_shapes()
    n_shapes = len(shapes)
    print(f"Noise shapes: {n_shapes}, amplitudes: {len(AMPLITUDES)}")

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    axes = axes.flatten()

    amp_norm = LogNorm(vmin=AMPLITUDES.min(), vmax=AMPLITUDES.max())
    amp_cmap = plt.cm.inferno

    # Assign a stable color to each shape
    shape_colors = plt.cm.tab20(np.linspace(0, 1, n_shapes))

    for ax_idx, omega_rabi in enumerate(OMEGAS):
        ax = axes[ax_idx]
        omega_label = f'{omega_rabi/np.pi:.0f}\\pi'

        seq_inst, seq_cont, tau_inst, tau_cont = build_cpmg_pair(omega_rabi)

        if seq_cont is None:
            ax.text(0.5, 0.5,
                    f'CPMG-{N_CPMG} does not fit\n'
                    f'$\\Omega = {omega_label}$',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14)
            ax.set_title(f'$\\Omega = {omega_label}$', fontsize=12)
            continue

        # CPMG resonance frequencies
        omega_res_inst = N_CPMG * np.pi / tau_inst
        omega_res_cont = N_CPMG * np.pi / tau_cont

        print(f"\n  Omega = {omega_label} "
              f"(tau_inst={tau_inst:.2f}, tau_cont={tau_cont:.2f})")
        print(f"    CPMG resonance: inst={omega_res_inst:.2f}, "
              f"cont={omega_res_cont:.2f}")

        all_vals = []
        n_inst_wins = 0

        for s_idx, (label, S_func, marker, family) in enumerate(shapes):
            dsy_inst_base = compute_delta_sy_sq(seq_inst, frequencies, S_func)
            dsy_cont_base = compute_delta_sy_sq(seq_cont, frequencies, S_func)

            ratio = dsy_inst_base / dsy_cont_base if dsy_cont_base > 0 else float('inf')
            side = 'INST' if ratio < 1 else 'cont'
            if ratio < 1:
                n_inst_wins += 1
            print(f"    {label:25s}  ratio={ratio:8.2f}x  [{side}]")

            dsy_i_arr = AMPLITUDES * dsy_inst_base
            dsy_c_arr = AMPLITUDES * dsy_cont_base
            all_vals.extend(dsy_i_arr)
            all_vals.extend(dsy_c_arr)

            sc = shape_colors[s_idx]

            # Track line
            ax.plot(dsy_i_arr, dsy_c_arr, color=sc, linewidth=1.8,
                    alpha=0.35, zorder=2)

            # Scatter colored by amplitude
            for A, di, dc in zip(AMPLITUDES, dsy_i_arr, dsy_c_arr):
                ax.scatter(di, dc, c=[amp_cmap(amp_norm(A))],
                           marker=marker, s=28, edgecolors='none',
                           zorder=3, alpha=0.9)

            # Label at high-A end
            ax.annotate(label, (dsy_i_arr[-1], dsy_c_arr[-1]),
                        fontsize=5.5, color=sc, fontweight='bold',
                        alpha=0.85, xytext=(4, 0),
                        textcoords='offset points', va='center')

        print(f"    => {n_inst_wins}/{n_shapes} noise shapes favor instantaneous")

        # Diagonal and shading
        all_pos = [v for v in all_vals if v > 0]
        if all_pos:
            vmin, vmax = min(all_pos) * 0.3, max(all_pos) * 5
            diag = np.array([vmin, vmax])
            ax.plot(diag, diag, 'k-', linewidth=1.2, alpha=0.4, zorder=1)
            ax.fill_between(diag, vmin, diag,
                            alpha=0.05, color='#2ca02c', zorder=0)
            ax.fill_between(diag, diag, vmax,
                            alpha=0.05, color='#1f77b4', zorder=0)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_aspect('equal')
        ax.set_xlabel(
            r'$\langle\delta\sigma_y^2\rangle_{\rm inst}$', fontsize=11)
        ax.set_ylabel(
            r'$\langle\delta\sigma_y^2\rangle_{\rm cont}$', fontsize=11)
        ax.set_title(
            f'CPMG-{N_CPMG},  $\\Omega = {omega_label}$\n'
            f'$\\tau_{{\\rm free}}$: inst={tau_inst:.1f}, cont={tau_cont:.1f}',
            fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.12, which='both')

        ax.text(0.95, 0.05, 'Continuous\nbetter', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='#2ca02c',
                alpha=0.6, fontstyle='italic')
        ax.text(0.05, 0.95, 'Instantaneous\nbetter', transform=ax.transAxes,
                ha='left', va='top', fontsize=8, color='#1f77b4',
                alpha=0.6, fontstyle='italic')

    # Family legend
    family_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
               markeredgecolor='k', markersize=7, label='Power-law'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='gray',
               markeredgecolor='k', markersize=7, label='Lorentzian'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='gray',
               markeredgecolor='k', markersize=7, label='Peaked'),
    ]
    fig.legend(handles=family_handles, loc='lower center', ncol=3,
               fontsize=10, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.42, -0.01))

    # Colorbar
    sm = ScalarMappable(cmap=amp_cmap, norm=amp_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Noise amplitude $A$', fontsize=11)

    fig.suptitle(
        r'$\langle\delta\sigma_y^2\rangle$: Instantaneous vs Continuous '
        f'CPMG-{N_CPMG}\n'
        f'Fixed wall-clock $T = {T_WALL:.2f}$, $\\delta = {DELTA}$',
        fontsize=13, y=1.01
    )

    plt.tight_layout(rect=[0, 0.03, 0.90, 0.96])

    if SAVE_FIGURE:
        for ext in ['pdf', 'png']:
            path = fig_dir / f'continuous_vs_instantaneous_performance.{ext}'
            fig.savefig(path, dpi=DPI, bbox_inches='tight')
            print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
