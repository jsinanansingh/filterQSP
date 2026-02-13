"""
Protocol performance comparison: noise susceptibility across noise environments.

Compares instantaneous CPMG, continuous CPMG, and GPS (continuous only)
at fixed total wall-clock time T.  Two figures of merit:

1. Total susceptibility (all Bloch components):
    chi = (2/pi) int S(w) (|Fx|^2 + |Fy|^2 + |Fz|^2) dw

2. sigma_y measurement susceptibility (decoherence of <sigma_y>):
    chi_y = (2/pi) int S(w) (|Fx|^2 + |Fz|^2) dw

   This measures noise that rotates the Bloch vector OUT of the y-axis,
   i.e. the components perpendicular to the measurement observable.

computed via the FFT matrix filter function.  Lower chi = better noise
filtering.

Layout: one panel per noise PSD (2x4 grid).
Each panel:
  - X-axis: protocol (categorical - each protocol is its own position)
  - Y-axis: chi (log scale)
  - Vertical line connecting 5 amplitude dots at each protocol position
  - Line color = protocol family  (blue=inst, red=cont, green=GPS)
  - Marker shape = noise PSD family (same within each panel)
  - Marker fill  = noise amplitude  (5 levels, colorbar)

Usage:
    python scripts/plot_qubit_performance.py

Output:
    figures/qubit_performance_plots/protocol_performance.{pdf,png}
    figures/qubit_performance_plots/protocol_performance_sy.{pdf,png}
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
    SIGMA_Z,
    ramsey_sequence,
    cpmg_sequence,
    continuous_ramsey_sequence,
    continuous_cpmg_sequence,
    ContinuousPulseSequence,
    ColoredNoisePSD,
    fft_filter_function,
    noise_susceptibility_from_matrix,
    bloch_components_from_matrix,
)

# =============================================================================
# Configuration
# =============================================================================

T_WALL = 2 * np.pi          # Fixed total wall-clock time

N_FFT = 4096                 # FFT samples (power of 2)
PAD_FACTOR = 4               # Zero-padding factor

H_NOISE = SIGMA_Z / 2       # Noise Hamiltonian

# Noise amplitudes: 5 levels from 10^-1 to 10^2
AMPLITUDES = np.logspace(-1, 2, 5)

# Noise PSD definitions: (label, kind, param, marker, family_name)
NOISE_DEFS = [
    ('White',                      'pl',  0,    'o', 'Power-law'),
    ('$1/f$',                      'pl',  1,    'o', 'Power-law'),
    ('$1/f^2$',                    'pl',  2,    'o', 'Power-law'),
    (r'Lor $\gamma\!=0.1$',       'lor', 0.1,  's', 'Lorentzian'),
    (r'Lor $\gamma\!=10$',        'lor', 10.,  's', 'Lorentzian'),
    (r'Peak $\omega_0\!=0.1$',    'pk',  0.1,  'D', 'Peaked'),
    (r'Peak $\omega_0\!=1$',      'pk',  1.0,  'D', 'Peaked'),
    (r'Peak $\omega_0\!=10$',     'pk',  10.,  'D', 'Peaked'),
]

# Protocol definitions: (label, family, build_key)
# family determines color/style; build_key is (type, n)
PROTOCOLS = [
    ('Ram',    'inst', ('inst', 0)),
    ('C1',     'inst', ('inst', 1)),
    ('C2',     'inst', ('inst', 2)),
    ('C4',     'inst', ('inst', 4)),
    ('C8',     'inst', ('inst', 8)),
    ('Ram',    'cont', ('cont', 0)),
    ('C1',     'cont', ('cont', 1)),
    ('C2',     'cont', ('cont', 2)),
    ('C4',     'cont', ('cont', 4)),
    ('C8',     'cont', ('cont', 8)),
    ('G1',     'gps',  ('gps',  1)),
    ('G2',     'gps',  ('gps',  2)),
    ('G4',     'gps',  ('gps',  4)),
    ('G8',     'gps',  ('gps',  8)),
]

# Plot
DPI = 150
FONT_SIZE = 10
OUTPUT_DIR = 'figures/qubit_performance_plots'

# Protocol family colors
FAMILY_COLORS = {
    'inst': '#1f77b4',   # blue
    'cont': '#d62728',   # red
    'gps':  '#2ca02c',   # green
}

# =============================================================================
# Noise PSD builders
# =============================================================================


def build_noise_funcs():
    """Return list of (label, S_func, marker_char, family_name)."""
    shapes = []
    for label, kind, param, marker, family in NOISE_DEFS:
        if kind == 'pl':
            S = ColoredNoisePSD.generic_power_law(
                amplitude=1.0, exponent=param, cutoff=1e-10)
        elif kind == 'lor':
            S = ColoredNoisePSD.lorentzian(amplitude=1.0, gamma=param)
        else:
            w0 = param
            def make_pk(w0_=w0):
                def psd(w):
                    w = np.asarray(w)
                    return np.exp(-((w - w0_)**2) / (2*(0.3*w0_)**2))
                return psd
            S = make_pk()
        shapes.append((label, S, marker, family))
    return shapes


# =============================================================================
# Sequence builders (return a single sequence at delta=0)
# =============================================================================


def build_inst_sequence(n):
    """Build instantaneous Ramsey (n=0) or CPMG-n."""
    if n == 0:
        return ramsey_sequence(tau=T_WALL, delta=0.0)
    return cpmg_sequence(tau=T_WALL, n_pulses=n, delta=0.0)


def build_cont_sequence(n):
    """Build continuous Ramsey (n=0) or CPMG-n."""
    omega = 4 * np.pi * (n + 1) / T_WALL
    tau_free = 3 * T_WALL / 4
    if n == 0:
        return continuous_ramsey_sequence(
            omega=omega, tau=tau_free, delta=0.0)
    return continuous_cpmg_sequence(
        omega=omega, tau=tau_free, n_pulses=n, delta=0.0)


def build_gps_sequence(n_cycles):
    """Build GPS sequence with n_cycles."""
    OMEGA_FAST = 20 * np.pi
    tau_pi2 = np.pi / (2 * OMEGA_FAST)
    tau_drive = T_WALL - 2 * tau_pi2
    omega_drive = 2 * np.pi * n_cycles / tau_drive
    seq = ContinuousPulseSequence()
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_continuous_pulse(omega_drive, [1, 0, 0], 0.0, tau_drive)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    return seq


# =============================================================================
# Plotting
# =============================================================================


def plot_performance_figure(base_sens, noise_shapes, title_extra, ylabel,
                            out_dir, filename):
    """Generate a 2x4 performance grid figure.

    Parameters
    ----------
    base_sens : dict
        Mapping build_key -> array of chi values (one per noise shape).
    noise_shapes : list
        Output of build_noise_funcs().
    title_extra : str
        Extra text appended to the suptitle formula line.
    ylabel : str
        Y-axis label for left panels.
    out_dir : Path
        Output directory.
    filename : str
        Base filename (without extension).
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharey=False)
    axes = axes.flatten()

    amp_norm = LogNorm(vmin=AMPLITUDES.min(), vmax=AMPLITUDES.max())
    amp_cmap = plt.cm.inferno

    n_prot = len(PROTOCOLS)

    # Build x-axis positions with gaps between families
    x_positions = np.arange(n_prot, dtype=float)
    for i in range(n_prot):
        family = PROTOCOLS[i][1]
        if family == 'cont':
            x_positions[i] += 0.8
        elif family == 'gps':
            x_positions[i] += 1.6

    x_labels = [p[0] for p in PROTOCOLS]

    for noise_idx, (noise_label, _, noise_marker, _) in enumerate(noise_shapes):
        ax = axes[noise_idx]

        for p_idx, (p_label, p_family, p_key) in enumerate(PROTOCOLS):
            x = x_positions[p_idx]
            color = FAMILY_COLORS[p_family]
            base = base_sens[p_key][noise_idx]

            if not np.isfinite(base) or base <= 0:
                continue

            y_vals = AMPLITUDES * base

            ax.plot([x, x], [y_vals[0], y_vals[-1]],
                    color=color, lw=1.8, alpha=0.45, zorder=2,
                    solid_capstyle='round')

            for A, y in zip(AMPLITUDES, y_vals):
                fc = amp_cmap(amp_norm(A))
                ax.scatter(x, y, marker=noise_marker, s=45,
                           facecolor=fc, edgecolor=color, linewidth=0.9,
                           zorder=4, alpha=0.95)

        ax.set_yscale('log')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(noise_label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.15, which='major', axis='y')

        inst_center = np.mean(x_positions[:5])
        cont_center = np.mean(x_positions[5:10])
        gps_center = np.mean(x_positions[10:])
        y_label_pos = -0.18
        ax.text(inst_center, y_label_pos, 'Instantaneous',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color=FAMILY_COLORS['inst'],
                fontweight='bold')
        ax.text(cont_center, y_label_pos, 'Continuous',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color=FAMILY_COLORS['cont'],
                fontweight='bold')
        ax.text(gps_center, y_label_pos, 'GPS',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color=FAMILY_COLORS['gps'],
                fontweight='bold')

    for i in [0, 4]:
        axes[i].set_ylabel(ylabel, fontsize=11)

    # --- Legends ---
    family_handles = [
        Line2D([0], [0], color=FAMILY_COLORS['inst'], lw=2.5,
               label='Instantaneous CPMG'),
        Line2D([0], [0], color=FAMILY_COLORS['cont'], lw=2.5,
               label='Continuous CPMG'),
        Line2D([0], [0], color=FAMILY_COLORS['gps'], lw=2.5,
               label='GPS (continuous)'),
    ]
    shape_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
               markeredgecolor='k', markersize=7, label='Power-law'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='gray',
               markeredgecolor='k', markersize=7, label='Lorentzian'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='gray',
               markeredgecolor='k', markersize=7, label='Peaked'),
    ]

    fig.legend(handles=family_handles + shape_handles,
               loc='lower center', ncol=6, fontsize=9,
               frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.45, -0.01))

    # --- Amplitude colorbar (wider and more visible) ---
    sm = ScalarMappable(cmap=amp_cmap, norm=amp_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.025, 0.75])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Noise amplitude $A$', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        title_extra
        + f'\nFixed $T = {T_WALL:.2f}$, '
        + r'$\delta = 0$, '
        + r'continuous pulses fill $T/4$',
        fontsize=12, y=0.99)

    fig.subplots_adjust(
        left=0.05, right=0.90, bottom=0.12, top=0.90,
        wspace=0.30, hspace=0.45)

    for ext in ['pdf', 'png']:
        p = out_dir / f'{filename}.{ext}'
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print(f'Saved: {p}')

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main():
    rcParams.update({
        'font.size': FONT_SIZE,
        'axes.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    out_dir = Path(__file__).parent.parent / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_shapes = build_noise_funcs()

    # ------------------------------------------------------------------
    # Build sequences and compute FFT filter functions
    # ------------------------------------------------------------------
    print('Computing FFT filter functions...')
    # Store both total susceptibility and per-component
    fft_data = {}       # build_key -> (freqs, suscept_total)
    fft_data_sy = {}    # build_key -> (freqs, suscept_sy = |Fx|^2 + |Fz|^2)

    for _, _, (typ, n) in PROTOCOLS:
        key = (typ, n)
        if key in fft_data:
            continue

        if typ == 'inst':
            seq = build_inst_sequence(n)
        elif typ == 'cont':
            seq = build_cont_sequence(n)
        else:
            seq = build_gps_sequence(n)

        freqs, F_mat = fft_filter_function(
            seq, H_NOISE, n_samples=N_FFT, pad_factor=PAD_FACTOR)

        suscept_total = noise_susceptibility_from_matrix(F_mat)
        fft_data[key] = (freqs, suscept_total)

        # Bloch components for sigma_y measurement
        Fx, Fy, Fz = bloch_components_from_matrix(F_mat)
        suscept_sy = np.abs(Fx)**2 + np.abs(Fz)**2
        fft_data_sy[key] = (freqs, suscept_sy)

        print(f'  {str(key):<20s}  T={seq.total_duration():.4f}  '
              f'freq range [{freqs[0]:.2f}, {freqs[-1]:.1f}]  '
              f'max|F|^2={np.max(suscept_total):.4e}  '
              f'max|F_perp_y|^2={np.max(suscept_sy):.4e}')

    # ------------------------------------------------------------------
    # Integrated noise susceptibility for each (protocol, noise) pair
    # ------------------------------------------------------------------
    print('Integrating noise susceptibilities...')
    print(f'  {"Protocol":<20s} | {"Noise":<25s} | {"chi_total":>12s} | {"chi_sy":>12s}')
    print(f'  {"-"*20} | {"-"*25} | {"-"*12} | {"-"*12}')

    base_sens = {}
    base_sens_sy = {}
    for key in fft_data:
        freqs, suscept = fft_data[key]
        _, suscept_sy = fft_data_sy[key]
        chi_vals = np.zeros(len(noise_shapes))
        chi_vals_sy = np.zeros(len(noise_shapes))
        for ni, (noise_label, S_func, _, _) in enumerate(noise_shapes):
            S_vals = S_func(freqs)
            chi = 2 * simpson(S_vals * suscept, x=freqs) / np.pi
            chi_sy = 2 * simpson(S_vals * suscept_sy, x=freqs) / np.pi
            chi_vals[ni] = chi
            chi_vals_sy[ni] = chi_sy
            print(f'  {str(key):<20s} | {noise_label:<25s} | {chi:12.4e} | {chi_sy:12.4e}')
        base_sens[key] = chi_vals
        base_sens_sy[key] = chi_vals_sy

    # ------------------------------------------------------------------
    # Figure 1: Total susceptibility
    # ------------------------------------------------------------------
    print('Plotting total susceptibility...')
    plot_performance_figure(
        base_sens, noise_shapes,
        title_extra=(
            r'Protocol Performance: $\chi = \frac{2}{\pi}'
            r'\int S(\omega)\,|F(\omega)|^2\,d\omega$'
            r' across noise environments'),
        ylabel=r'$\chi$ (noise susceptibility)',
        out_dir=out_dir,
        filename='protocol_performance',
    )

    # ------------------------------------------------------------------
    # Figure 2: sigma_y measurement susceptibility
    # ------------------------------------------------------------------
    print('Plotting sigma_y measurement susceptibility...')
    plot_performance_figure(
        base_sens_sy, noise_shapes,
        title_extra=(
            r'$\sigma_y$ Measurement: $\chi_y = \frac{2}{\pi}'
            r'\int S(\omega)\,(|F_x|^2 + |F_z|^2)\,d\omega$'
            r' across noise environments'),
        ylabel=r'$\chi_y$ ($\sigma_y$ decoherence)',
        out_dir=out_dir,
        filename='protocol_performance_sy',
    )


if __name__ == '__main__':
    main()
