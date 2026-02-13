"""
Detuning optimization for all protocol types.

For each (protocol, noise PSD) pair, finds the optimal detuning delta that
minimizes the Cramer-Rao frequency estimation variance, then compares
performance at delta=0 vs optimized delta.

Produces:
  - Console summary table
  - Figure 1: paired comparison (delta=0 vs optimized) per noise environment
  - Figure 2: improvement ratio bar charts

Usage:
    python scripts/optimize_protocol_detuning.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
)
from quantum_pulse_suite.analysis import optimize_detuning

# =============================================================================
# Configuration (reused from plot_qubit_performance.py)
# =============================================================================

T_WALL = 2 * np.pi          # Fixed total wall-clock time
N_FFT = 4096                 # FFT samples (power of 2)
PAD_FACTOR = 4               # Zero-padding factor
H_NOISE = SIGMA_Z / 2       # Noise Hamiltonian

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

FAMILY_COLORS = {
    'inst': '#1f77b4',   # blue
    'cont': '#d62728',   # red
    'gps':  '#2ca02c',   # green
}

DPI = 150
FONT_SIZE = 10
OUTPUT_DIR = 'figures/qubit_performance_plots'

# Optimization parameters
DELTA_RANGE = (0.01, 5.0)
N_GRID = 200
OPT_FREQS = np.linspace(0.01, 200, 2000)

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
# Sequence builder factories (delta -> PulseSequence closures)
# =============================================================================


def make_inst_builder(n):
    """Return delta -> InstantaneousPulseSequence for Ramsey (n=0) or CPMG-n."""
    if n == 0:
        return lambda delta: ramsey_sequence(tau=T_WALL, delta=delta)
    return lambda delta: cpmg_sequence(tau=T_WALL, n_pulses=n, delta=delta)


def make_cont_builder(n):
    """Return delta -> ContinuousPulseSequence for Ramsey (n=0) or CPMG-n."""
    omega = 4 * np.pi * (n + 1) / T_WALL
    tau_free = 3 * T_WALL / 4
    if n == 0:
        return lambda delta: continuous_ramsey_sequence(
            omega=omega, tau=tau_free, delta=delta)
    return lambda delta: continuous_cpmg_sequence(
        omega=omega, tau=tau_free, n_pulses=n, delta=delta)


def make_gps_builder(n_cycles):
    """Return delta -> ContinuousPulseSequence for GPS with n_cycles."""
    OMEGA_FAST = 20 * np.pi
    tau_pi2 = np.pi / (2 * OMEGA_FAST)
    tau_drive = T_WALL - 2 * tau_pi2
    omega_drive = 2 * np.pi * n_cycles / tau_drive

    def builder(delta):
        seq = ContinuousPulseSequence()
        seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], delta, tau_pi2)
        seq.add_continuous_pulse(omega_drive, [1, 0, 0], delta, tau_drive)
        seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], delta, tau_pi2)
        return seq
    return builder


def get_seq_builder(typ, n):
    """Dispatch to the appropriate builder factory."""
    if typ == 'inst':
        return make_inst_builder(n)
    elif typ == 'cont':
        return make_cont_builder(n)
    else:
        return make_gps_builder(n)


# =============================================================================
# Compute FFT chi for a sequence at a given delta
# =============================================================================


def compute_chi(seq, S_func):
    """Compute total noise susceptibility chi via FFT filter function."""
    freqs, F_mat = fft_filter_function(
        seq, H_NOISE, n_samples=N_FFT, pad_factor=PAD_FACTOR)
    suscept = noise_susceptibility_from_matrix(F_mat)
    S_vals = S_func(freqs)
    return float(2 * simpson(S_vals * suscept, x=freqs) / np.pi)


# =============================================================================
# Plotting
# =============================================================================


def plot_comparison_figure(chi_baseline, chi_optimized, delta_opts,
                           noise_shapes, out_dir):
    """Figure 1: paired markers showing delta=0 vs optimized delta."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharey=False)
    axes = axes.flatten()

    n_prot = len(PROTOCOLS)
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

            chi0 = chi_baseline[p_key][noise_idx]
            chi_opt = chi_optimized[p_key][noise_idx]
            d_opt = delta_opts[p_key][noise_idx]

            if not np.isfinite(chi0) or chi0 <= 0:
                continue

            # Hollow marker for delta=0
            ax.scatter(x - 0.12, chi0, marker=noise_marker, s=55,
                       facecolor='none', edgecolor=color, linewidth=1.2,
                       zorder=4, alpha=0.9)

            if np.isfinite(chi_opt) and chi_opt > 0:
                # Filled marker for optimized delta
                ax.scatter(x + 0.12, chi_opt, marker=noise_marker, s=55,
                           facecolor=color, edgecolor=color, linewidth=1.0,
                           zorder=5, alpha=0.9)

                # Arrow connecting them
                ax.annotate('', xy=(x + 0.10, chi_opt),
                            xytext=(x - 0.10, chi0),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=1.0, alpha=0.5))

        ax.set_yscale('log')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(noise_label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.15, which='major', axis='y')

        # Family labels below x-axis
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
        axes[i].set_ylabel(r'$\chi$ (noise susceptibility)', fontsize=11)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor='gray', markersize=8, linewidth=0,
               label=r'$\delta = 0$'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
               markeredgecolor='gray', markersize=8, linewidth=0,
               label=r'Optimized $\delta$'),
        Line2D([0], [0], color=FAMILY_COLORS['inst'], lw=2.5,
               label='Instantaneous CPMG'),
        Line2D([0], [0], color=FAMILY_COLORS['cont'], lw=2.5,
               label='Continuous CPMG'),
        Line2D([0], [0], color=FAMILY_COLORS['gps'], lw=2.5,
               label='GPS (continuous)'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=5, fontsize=9,
               frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.50, -0.01))

    fig.suptitle(
        r'Detuning Optimization: $\chi(\delta=0)$ vs $\chi(\delta_{\rm opt})$'
        f'\nFixed $T = {T_WALL:.2f}$',
        fontsize=12, y=0.99)

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.12, top=0.90,
        wspace=0.30, hspace=0.45)

    for ext in ['pdf', 'png']:
        p = out_dir / f'detuning_optimization_comparison.{ext}'
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)


def plot_ratio_figure(chi_baseline, chi_optimized, noise_shapes, out_dir):
    """Figure 2: bar charts of improvement ratio chi(d=0)/chi(d_opt)."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharey=False)
    axes = axes.flatten()

    n_prot = len(PROTOCOLS)
    x_positions = np.arange(n_prot, dtype=float)
    for i in range(n_prot):
        family = PROTOCOLS[i][1]
        if family == 'cont':
            x_positions[i] += 0.8
        elif family == 'gps':
            x_positions[i] += 1.6

    x_labels = [p[0] for p in PROTOCOLS]

    for noise_idx, (noise_label, _, _, _) in enumerate(noise_shapes):
        ax = axes[noise_idx]

        for p_idx, (p_label, p_family, p_key) in enumerate(PROTOCOLS):
            x = x_positions[p_idx]
            color = FAMILY_COLORS[p_family]

            chi0 = chi_baseline[p_key][noise_idx]
            chi_opt = chi_optimized[p_key][noise_idx]

            if (not np.isfinite(chi0) or chi0 <= 0
                    or not np.isfinite(chi_opt) or chi_opt <= 0):
                continue

            ratio = chi0 / chi_opt
            ax.bar(x, ratio, width=0.6, color=color, alpha=0.75,
                   edgecolor='k', linewidth=0.5)

        # Horizontal line at ratio=1
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(noise_label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.15, which='major', axis='y')

        # Family labels below x-axis
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
        axes[i].set_ylabel(r'Improvement ratio $\chi(0) / \chi(\delta_{\rm opt})$',
                           fontsize=11)

    legend_handles = [
        Line2D([0], [0], color=FAMILY_COLORS['inst'], lw=8,
               label='Instantaneous CPMG'),
        Line2D([0], [0], color=FAMILY_COLORS['cont'], lw=8,
               label='Continuous CPMG'),
        Line2D([0], [0], color=FAMILY_COLORS['gps'], lw=8,
               label='GPS (continuous)'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=3, fontsize=9,
               frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.50, -0.01))

    fig.suptitle(
        r'Improvement from Detuning Optimization: $\chi(\delta=0)\,/\,\chi(\delta_{\rm opt})$'
        f'\nFixed $T = {T_WALL:.2f}$, ratio > 1 means optimization helps',
        fontsize=12, y=0.99)

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.12, top=0.90,
        wspace=0.30, hspace=0.45)

    for ext in ['pdf', 'png']:
        p = out_dir / f'detuning_optimization_ratio.{ext}'
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

    n_noise = len(noise_shapes)

    # Collect unique build keys
    unique_keys = []
    seen = set()
    for _, _, key in PROTOCOLS:
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)

    # ------------------------------------------------------------------
    # Step 1: Baseline chi at delta=0
    # ------------------------------------------------------------------
    print('=' * 72)
    print('Step 1: Computing baseline chi at delta=0 ...')
    print('=' * 72)

    chi_baseline = {}
    for typ, n in unique_keys:
        key = (typ, n)
        builder = get_seq_builder(typ, n)
        seq = builder(0.0)

        freqs, F_mat = fft_filter_function(
            seq, H_NOISE, n_samples=N_FFT, pad_factor=PAD_FACTOR)
        suscept = noise_susceptibility_from_matrix(F_mat)

        chi_vals = np.zeros(n_noise)
        for ni, (noise_label, S_func, _, _) in enumerate(noise_shapes):
            S_vals = S_func(freqs)
            chi_vals[ni] = float(2 * simpson(S_vals * suscept, x=freqs) / np.pi)

        chi_baseline[key] = chi_vals
        print(f'  {str(key):<20s}  chi range: [{chi_vals.min():.4e}, {chi_vals.max():.4e}]')

    # ------------------------------------------------------------------
    # Step 2: Optimize detuning for each (protocol, noise) pair
    # ------------------------------------------------------------------
    print()
    print('=' * 72)
    print('Step 2: Optimizing detuning (this may take a few minutes) ...')
    print('=' * 72)

    delta_opts = {}    # key -> array of optimal deltas
    var_opts = {}      # key -> array of optimal variances

    total = len(unique_keys) * n_noise
    count = 0

    for typ, n in unique_keys:
        key = (typ, n)
        builder = get_seq_builder(typ, n)

        d_arr = np.zeros(n_noise)
        v_arr = np.full(n_noise, np.inf)

        for ni, (noise_label, S_func, _, _) in enumerate(noise_shapes):
            count += 1
            print(f'  [{count:3d}/{total}] {str(key):<20s} | {noise_label:<25s} ... ',
                  end='', flush=True)

            result = optimize_detuning(
                builder, OPT_FREQS, S_func, B_lab=None,
                delta_range=DELTA_RANGE, n_grid=N_GRID)

            d_arr[ni] = result.delta_opt
            v_arr[ni] = result.min_variance
            print(f'delta_opt={result.delta_opt:.4f}  var={result.min_variance:.4e}')

        delta_opts[key] = d_arr
        var_opts[key] = v_arr

    # ------------------------------------------------------------------
    # Step 3: Recompute chi at optimized delta
    # ------------------------------------------------------------------
    print()
    print('=' * 72)
    print('Step 3: Recomputing chi at optimized deltas ...')
    print('=' * 72)

    chi_optimized = {}
    for typ, n in unique_keys:
        key = (typ, n)
        builder = get_seq_builder(typ, n)
        chi_vals = np.zeros(n_noise)

        for ni, (noise_label, S_func, _, _) in enumerate(noise_shapes):
            d_opt = delta_opts[key][ni]
            seq = builder(d_opt)
            chi_vals[ni] = compute_chi(seq, S_func)

        chi_optimized[key] = chi_vals
        print(f'  {str(key):<20s}  chi_opt range: [{chi_vals.min():.4e}, {chi_vals.max():.4e}]')

    # ------------------------------------------------------------------
    # Step 4: Console summary table
    # ------------------------------------------------------------------
    print()
    print('=' * 72)
    print('Summary Table')
    print('=' * 72)
    header = (f'  {"Protocol":<20s} | {"Noise":<25s} | {"delta_opt":>10s} | '
              f'{"chi(d=0)":>12s} | {"chi(d_opt)":>12s} | {"Ratio":>8s} | '
              f'{"var_opt":>12s}')
    print(header)
    print(f'  {"-"*20} | {"-"*25} | {"-"*10} | {"-"*12} | {"-"*12} | {"-"*8} | {"-"*12}')

    for typ, n in unique_keys:
        key = (typ, n)
        for ni, (noise_label, _, _, _) in enumerate(noise_shapes):
            d_opt = delta_opts[key][ni]
            chi0 = chi_baseline[key][ni]
            chi_opt = chi_optimized[key][ni]
            v_opt = var_opts[key][ni]

            if np.isfinite(chi_opt) and chi_opt > 0 and chi0 > 0:
                ratio = chi0 / chi_opt
                ratio_str = f'{ratio:8.3f}'
            else:
                ratio_str = '     N/A'

            if np.isinf(v_opt):
                v_str = '         inf'
                d_str = '       N/A'
            else:
                v_str = f'{v_opt:12.4e}'
                d_str = f'{d_opt:10.4f}'

            print(f'  {str(key):<20s} | {noise_label:<25s} | {d_str} | '
                  f'{chi0:12.4e} | {chi_opt:12.4e} | {ratio_str} | {v_str}')

    # ------------------------------------------------------------------
    # Step 5: Figures
    # ------------------------------------------------------------------
    print()
    print('=' * 72)
    print('Generating figures ...')
    print('=' * 72)

    plot_comparison_figure(chi_baseline, chi_optimized, delta_opts,
                           noise_shapes, out_dir)
    plot_ratio_figure(chi_baseline, chi_optimized, noise_shapes, out_dir)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
