"""
High-frequency noise analysis for three-level clock protocols.

Models two realistic atomic-clock noise scenarios:

  1. High-pass white noise  S(ω) = θ(ω − ω_c)
     Sweeps ω_c from 0 to 30 rad/s to show how FOM changes as low-frequency
     noise is eliminated.  Models a laser whose 1/f corner is below ω_c (only
     white noise remains above the corner).

  2. Bandpass / peaked noise  S(ω) = Lorentzian centred at ω_peak
     Models a sharp technical noise source (power-line harmonic, servo bump,
     vibration) at a specific frequency.  GPS m=8 has nulls at ω = 8k; if the
     peak falls on one of those nulls the GPS protocol suppresses it entirely.

Both figures share the same four protocols as the main protocol comparison.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
from quantum_pulse_suite.core.three_level_filter import (
    fft_three_level_filter,
    detuning_sensitivity,
)
from quantum_pulse_suite.analysis.pulse_optimizer import optimize_equiangular_sequence

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi
OMEGA_GPS1 = 2 * np.pi * 1 / T   # = 1.0
OMEGA_GPS8 = 2 * np.pi * 8 / T   # = 8.0
N_FFT      = 4 * 4096
PAD_FACTOR = 16   # extra padding for better high-freq resolution

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'


# =============================================================================
# Sequence builders (same as protocol comparison)
# =============================================================================

def build_ramsey(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


def build_gps(system, omega):
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, T)
    seq.compute_polynomials()
    return seq


def build_equiangular(system, omega, phases):
    tau = T / len(phases)
    seq = MultiLevelPulseSequence(system, system.probe)
    for phi in phases:
        seq.add_continuous_pulse(omega, [np.cos(phi), np.sin(phi), 0.0], 0.0, tau)
    seq.compute_polynomials()
    return seq


# =============================================================================
# FOM helpers
# =============================================================================

def fom_highpass(freqs, Fe, sens_sq, cutoff):
    """FOM with S(ω) = 1 for ω ≥ cutoff, else 0."""
    mask = freqs >= cutoff
    if mask.sum() < 2:
        return np.inf
    kubo = float(simpson(Fe[mask], x=freqs[mask]) / (2 * np.pi))
    return sens_sq / kubo if kubo > 0 else np.inf


def fom_lorentzian(freqs, Fe, sens_sq, omega_peak, width=0.5):
    """FOM with a Lorentzian noise peak: S(ω) = 1/((ω-ω_peak)^2 + width^2)."""
    S = 1.0 / ((freqs - omega_peak)**2 + width**2)
    kubo = float(simpson(Fe * S, x=freqs) / (2 * np.pi))
    return sens_sq / kubo if kubo > 0 else np.inf


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    # ── Build sequences ───────────────────────────────────────────────────────
    print('Building Ramsey ...')
    seq_R = build_ramsey(system)

    print('Building GPS m=1 (delta=0) ...')
    seq_G1 = build_gps(system, OMEGA_GPS1)

    print('Building GPS m=8 (delta=0) ...')
    seq_G8 = build_gps(system, OMEGA_GPS8)

    print('Optimising equiangular N=4 (white noise, seed=7) ...')
    res_eq4   = optimize_equiangular_sequence(
        system, T, N=4, noise_psd='white',
        n_restarts=12, seed=7, n_fft=2048, pad_factor=4,
        popsize=20, maxiter=500,
    )
    omega_eq4  = res_eq4.omega
    phases_eq4 = res_eq4.phases
    seq_eq4    = res_eq4.seq
    print(f'  Omega*T = {omega_eq4 * T:.4f},  phases = '
          f'{np.array2string(phases_eq4, precision=3, separator=", ")}')

    # ── Compute filter functions (high-res) ───────────────────────────────────
    print('\nComputing filter functions ...')
    protocols = [
        ('Ramsey',          seq_R,  'C3', '-'),
        ('GPS $m{=}1$',     seq_G1, 'C0', '-'),
        ('GPS $m{=}8$',     seq_G8, 'C2', '-'),
        ('Equiangular $N{=}4$', seq_eq4, 'C1', '-'),
    ]

    data = {}
    for label, seq, color, ls in protocols:
        _, sens_sq = detuning_sensitivity(seq)
        freqs, Fe, _, _ = fft_three_level_filter(
            seq, n_samples=N_FFT, pad_factor=PAD_FACTOR, m_y=1.0)
        data[label] = dict(sens_sq=sens_sq, freqs=freqs, Fe=Fe,
                           color=color, ls=ls)
        kubo_w = float(simpson(Fe, x=freqs) / (2 * np.pi))
        print(f'  {label:<26}  sens={sens_sq:.4f}  kubo_w={kubo_w:.4f}'
              f'  FOM_w={sens_sq/kubo_w:.2f}')

    # =========================================================================
    # Figure 1: FOM vs high-pass cutoff  ω_c
    # =========================================================================
    cutoffs = np.linspace(0.0, 28.0, 500)

    fig1, ax1 = plt.subplots(figsize=(8.5, 5.5))

    for label, _, color, ls in protocols:
        d = data[label]
        foms = [fom_highpass(d['freqs'], d['Fe'], d['sens_sq'], wc)
                for wc in cutoffs]
        foms = np.array(foms)
        # Clip very large values for legibility
        foms = np.clip(foms, 0, 1000)
        ax1.semilogy(cutoffs, foms, color=color, lw=2, ls=ls, label=label)

    # Mark GPS m=8 harmonic nulls
    for k in range(1, 5):
        wk = OMEGA_GPS8 * k
        if wk <= 28:
            ax1.axvline(wk, color='C2', lw=0.8, ls=':', alpha=0.5)
            ax1.text(wk + 0.15, 2.0, f'$8{k}$', color='C2',
                     fontsize=7, va='bottom')

    ax1.set_xlabel(r'Noise cutoff $\omega_c$ (rad s$^{-1}$)', fontsize=12)
    ax1.set_ylabel(r'FOM $= |\partial_\delta S|^2 / \mathcal{K}(\omega_c)$',
                   fontsize=12)
    ax1.set_title(
        r'FOM under high-pass white noise $S(\omega) = \theta(\omega - \omega_c)$'
        '\n'
        r'All protocols at $\delta = 0$, $T = 2\pi$',
        fontsize=11)
    ax1.set_xlim([0, 28])
    ax1.set_ylim([5, 1000])
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=9)
    fig1.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'hf_noise_fom_vs_cutoff.{ext}'
        fig1.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    # =========================================================================
    # Figure 2: FOM vs Lorentzian noise peak position ω_peak
    # =========================================================================
    omega_peaks = np.linspace(0.1, 28.0, 500)
    width = 0.3  # Lorentzian half-width

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))

    for label, _, color, ls in protocols:
        d = data[label]
        foms = [fom_lorentzian(d['freqs'], d['Fe'], d['sens_sq'],
                               wp, width=width)
                for wp in omega_peaks]
        foms = np.clip(np.array(foms), 0, 5000)
        ax2.semilogy(omega_peaks, foms, color=color, lw=2, ls=ls, label=label)

    # Mark GPS m=8 harmonics
    for k in range(1, 5):
        wk = OMEGA_GPS8 * k
        if wk <= 28:
            ax2.axvline(wk, color='C2', lw=0.8, ls=':', alpha=0.5)

    # Mark GPS m=1 harmonics
    for k in range(1, 10):
        wk = OMEGA_GPS1 * k
        if wk <= 28:
            ax2.axvline(wk, color='C0', lw=0.5, ls=':', alpha=0.3)

    ax2.set_xlabel(r'Noise peak $\omega_{\rm peak}$ (rad s$^{-1}$)', fontsize=12)
    ax2.set_ylabel(r'FOM $= |\partial_\delta S|^2 / \mathcal{K}(S_{\rm Lorentz})$',
                   fontsize=12)
    ax2.set_title(
        rf'FOM under a Lorentzian noise peak (half-width $\Gamma = {width}$)'
        '\n'
        r'All protocols at $\delta = 0$, $T = 2\pi$'
        '\n'
        r'Dotted lines: GPS $m{=}8$ harmonics (green), GPS $m{=}1$ harmonics (blue)',
        fontsize=10)
    ax2.set_xlim([0, 28])
    ax2.set_ylim([1, 5000])
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=9)
    fig2.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'hf_noise_lorentzian.{ext}'
        fig2.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')

    # ── Print FOM table at key cutoffs ────────────────────────────────────────
    print('\n--- FOM under high-pass white noise at key cutoffs ---')
    key_cutoffs = [0.0, 1.0, 4.0, 8.0, 16.0]
    hdr = f'{"Protocol":<26} ' + ' '.join(f'ω_c={wc:>4.0f}' for wc in key_cutoffs)
    print(hdr)
    print('-' * len(hdr))
    for label, _, _, _ in protocols:
        d = data[label]
        vals = [fom_highpass(d['freqs'], d['Fe'], d['sens_sq'], wc)
                for wc in key_cutoffs]
        row = f'{label:<26} ' + ' '.join(f'{v:>10.2f}' for v in vals)
        print(row)

    print('\n--- FOM under Lorentzian noise at GPS m=8 harmonics ---')
    hdr2 = f'{"Protocol":<26} ' + ' '.join(f'ω={wk:>4.0f}' for wk in [8, 16, 24])
    print(hdr2)
    print('-' * len(hdr2))
    for label, _, _, _ in protocols:
        d = data[label]
        vals = [fom_lorentzian(d['freqs'], d['Fe'], d['sens_sq'],
                               float(wk), width=width)
                for wk in [8, 16, 24]]
        row = f'{label:<26} ' + ' '.join(f'{v:>10.2f}' for v in vals)
        print(row)


if __name__ == '__main__':
    main()
