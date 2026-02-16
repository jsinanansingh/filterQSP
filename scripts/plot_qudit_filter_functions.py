"""
Qudit irrep-resolved filter functions for spin-displacement pulse sequences.

Produces figures showing:
  1. Irrep decomposition F_L(w) for qudit Ramsey (d=2,3,4,5)
  2. GGM component decomposition for d=3 Ramsey
  3. Transition-resolved decomposition for d=3
  4. Dimension scaling comparison (Ramsey & spin echo)

Usage:
    python scripts/plot_qudit_filter_functions.py

Output:
    figures/qudit_filter_functions/irrep_ramsey.{pdf,png}
    figures/qudit_filter_functions/ggm_components_d3.{pdf,png}
    figures/qudit_filter_functions/transition_resolved_d3.{pdf,png}
    figures/qudit_filter_functions/dimension_scaling.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.core.spin_displacement import (
    spin_j_operators,
    su2_scaling_factor,
)
from quantum_pulse_suite.core.qudit_pulse_sequence import QuditPulseSequence
from quantum_pulse_suite.core.fft_filter_function import (
    fft_filter_function,
    noise_susceptibility_from_matrix,
)
from quantum_pulse_suite.core.irrep_decomposition import (
    irrep_resolved_filter_function,
    transition_resolved_filter_function,
    ggm_component_filter_functions,
)

# =============================================================================
# Configuration
# =============================================================================

TAU = 2.0
N_SAMPLES = 8192

DPI = 150
FONT_SIZE = 11
OUTPUT_DIR = 'figures/qudit_filter_functions'

# Color palettes
IRREP_COLORS = ['#999999', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#ff7f00', '#a65628', '#f781bf', '#66c2a5']
DIM_COLORS = {2: '#1b9e77', 3: '#d95f02', 4: '#7570b3', 5: '#e7298a'}


def compute_qudit_filter(d, seq_type='ramsey', tau=TAU, n_pulses=1):
    """Build sequence and compute FFT filter function."""
    if seq_type == 'ramsey':
        seq = QuditPulseSequence.ramsey(d, tau=tau, continuous=False)
    elif seq_type == 'spin_echo':
        seq = QuditPulseSequence.spin_echo(d, tau=tau, continuous=False)
    elif seq_type == 'cpmg':
        seq = QuditPulseSequence.cpmg(d, tau=tau, n_pulses=n_pulses,
                                       continuous=False)
    else:
        raise ValueError(f"Unknown sequence type: {seq_type}")

    _, _, Jz = spin_j_operators(d)
    freqs, F_mat = fft_filter_function(seq, Jz, n_samples=N_SAMPLES)
    return freqs, F_mat


# =============================================================================
# Figure 1: Irrep decomposition for Ramsey across dimensions
# =============================================================================

def plot_irrep_ramsey():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, d in enumerate([2, 3, 4, 5]):
        ax = axes[idx]
        freqs, F_mat = compute_qudit_filter(d, 'ramsey')
        total = noise_susceptibility_from_matrix(F_mat)
        irrep_susc = irrep_resolved_filter_function(F_mat, d)

        mask = freqs > 0.5

        # Plot total
        ax.plot(freqs[mask], total[mask], 'k-', linewidth=2, alpha=0.4,
                label='Total', zorder=1)

        # Plot each irrep
        for L in sorted(irrep_susc.keys()):
            F_L = irrep_susc[L]
            color = IRREP_COLORS[L % len(IRREP_COLORS)]
            sig = F_L[mask] > 1e-15 * np.max(total[mask])
            if np.any(sig):
                ax.plot(freqs[mask], F_L[mask], color=color, linewidth=1.5,
                        label=f'$L={L}$', zorder=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$F_L(\omega)$')

        j = (d - 1) / 2
        ax.set_title(f'$d={d}$ (spin-$j={j:.0f}${"" if j == int(j) else f"/{2}"}$)$'
                      if j == int(j) else f'$d={d}$ (spin-${int(2*j)}/2$)',
                      fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
        ax.set_xlim(0.5, 200)
        ax.grid(True, alpha=0.12, which='both')

        ymin = max(1e-12, np.min(total[mask & (total > 0)]) * 0.1)
        ax.set_ylim(bottom=ymin)

    fig.suptitle(
        'Irrep-Resolved Filter Functions: Qudit Ramsey\n'
        r'$\mathrm{Jz}$ noise, instantaneous $J_x$ pulses, $\tau=' + f'{TAU}$',
        fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 2: GGM component decomposition for d=3
# =============================================================================

def plot_ggm_d3():
    d = 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (seq_type, title) in enumerate([('ramsey', 'Ramsey'),
                                              ('spin_echo', 'Spin Echo')]):
        ax = axes[idx]
        freqs, F_mat = compute_qudit_filter(d, seq_type)
        total = noise_susceptibility_from_matrix(F_mat)
        susc_comp, labels = ggm_component_filter_functions(F_mat, d)

        mask = freqs > 0.5

        # Plot total
        ax.plot(freqs[mask], total[mask], 'k-', linewidth=2.5, alpha=0.3,
                label='Total', zorder=1)

        # Plot each nonzero GGM component
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
        for a in range(len(labels)):
            power = np.sum(susc_comp[a, mask])
            if power > 1e-10 * np.sum(total[mask]):
                ax.plot(freqs[mask], susc_comp[a, mask], color=colors[a],
                        linewidth=1.2, alpha=0.8, label=f'$\\lambda_{{{labels[a]}}}$',
                        zorder=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$|R_a(\omega)|^2$')
        ax.set_title(f'{title} ($d={d}$)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='lower left', framealpha=0.9, ncol=2)
        ax.set_xlim(0.5, 200)
        ax.grid(True, alpha=0.12, which='both')

        ymin = max(1e-12, np.min(total[mask & (total > 0)]) * 0.1)
        ax.set_ylim(bottom=ymin)

    fig.suptitle(
        'GGM Component Filter Functions ($d=3$, Gell-Mann basis)\n'
        r'$J_z$ noise, instantaneous $J_x$ pulses',
        fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 3: Transition-resolved decomposition for d=3
# =============================================================================

def plot_transitions_d3():
    d = 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (seq_type, title) in enumerate([('ramsey', 'Ramsey'),
                                              ('spin_echo', 'Spin Echo')]):
        ax = axes[idx]
        freqs, F_mat = compute_qudit_filter(d, seq_type)
        total = noise_susceptibility_from_matrix(F_mat)
        trans_susc = transition_resolved_filter_function(F_mat, d)

        mask = freqs > 0.5

        ax.plot(freqs[mask], total[mask], 'k-', linewidth=2.5, alpha=0.3,
                label='Total', zorder=1)

        colors_trans = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        cidx = 0
        for key in sorted(trans_susc.keys(), key=str):
            F_t = trans_susc[key]
            power = np.sum(F_t[mask])
            if power > 1e-10 * np.sum(total[mask]):
                if key == ('diag',):
                    label = 'Diagonal'
                else:
                    label = f'$|{key[0]}\\rangle\\langle{key[1]}|$'
                ax.plot(freqs[mask], F_t[mask], color=colors_trans[cidx % 4],
                        linewidth=1.5, alpha=0.8, label=label, zorder=2)
                cidx += 1

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$F_{jk}(\omega)$')
        ax.set_title(f'{title} ($d={d}$)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
        ax.set_xlim(0.5, 200)
        ax.grid(True, alpha=0.12, which='both')

        ymin = max(1e-12, np.min(total[mask & (total > 0)]) * 0.1)
        ax.set_ylim(bottom=ymin)

    fig.suptitle(
        'Transition-Resolved Filter Functions ($d=3$)\n'
        r'$J_z$ noise, instantaneous $J_x$ pulses',
        fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# =============================================================================
# Figure 4: Dimension scaling comparison
# =============================================================================

def plot_dimension_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (seq_type, title) in enumerate([('ramsey', 'Ramsey'),
                                              ('spin_echo', 'Spin Echo'),
                                              ('cpmg', 'CPMG-2')]):
        ax = axes[idx]

        for d in [2, 3, 4, 5]:
            if seq_type == 'cpmg':
                freqs, F_mat = compute_qudit_filter(d, seq_type, n_pulses=2)
            else:
                freqs, F_mat = compute_qudit_filter(d, seq_type)

            total = noise_susceptibility_from_matrix(F_mat)
            scaling = su2_scaling_factor(d)

            mask = freqs > 0.5

            # Plot normalized susceptibility (divided by scaling factor)
            ax.plot(freqs[mask], total[mask] / scaling,
                    color=DIM_COLORS[d], linewidth=1.5, alpha=0.8,
                    label=f'$d={d}$ ($\\times {scaling:.0f}$)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$F(\omega) / [d(d^2{-}1)/6]$')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
        ax.set_xlim(0.5, 200)
        ax.grid(True, alpha=0.12, which='both')

    fig.suptitle(
        'SU(2) Embedding Scaling: Qudit vs Qubit Filter Functions\n'
        r'$F_d(\omega) / [d(d^2{-}1)/6]$ should overlap $\Rightarrow$ '
        r'all curves collapse onto qubit result',
        fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


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

    plots = [
        ('irrep_ramsey', plot_irrep_ramsey),
        ('ggm_components_d3', plot_ggm_d3),
        ('transition_resolved_d3', plot_transitions_d3),
        ('dimension_scaling', plot_dimension_scaling),
    ]

    for name, plot_func in plots:
        print(f'Generating {name}...')
        fig = plot_func()
        for ext in ['pdf', 'png']:
            p = out_dir / f'{name}.{ext}'
            fig.savefig(p, dpi=DPI, bbox_inches='tight')
            print(f'  Saved: {p}')
        plt.close(fig)

    print('Done.')


if __name__ == '__main__':
    main()
