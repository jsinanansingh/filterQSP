"""
Two-corner noise comparison: S(omega) = (1 + omega_c1/|omega|) * theta(omega_c2 - |omega|)

Sweeps omega_c2 with several fixed omega_c1 values, evaluating FOM for:
  - GPS m = 1, 2, 4, 8, 16
  - Equiangular N=4 optimised for each (omega_c1, omega_c2) noise

Produces:
  1. FOM vs omega_c2 curves for each protocol  (Figure 1, per omega_c1)
  2. Optimal equiangular Omega* vs omega_c2     (Figure 2)
  3. Summary table from point optimisations
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
    fft_three_level_filter, detuning_sensitivity,
)
from quantum_pulse_suite.analysis.pulse_optimizer import (
    optimize_equiangular_sequence, mixed_noise_psd,
)

T          = 2 * np.pi
N_FFT      = 4096
PAD_FACTOR = 4
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'


# =============================================================================
# Sequence builders
# =============================================================================

def build_gps(system, m):
    omega = 2 * np.pi * m / T
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, T)
    seq.compute_polynomials()
    return seq, omega


def build_ramsey(system):
    omega_fast = 20 * np.pi
    tau_pi2    = np.pi / (2 * omega_fast)
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega_fast, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(T - 2 * tau_pi2, 0.0)
    seq.add_continuous_pulse(omega_fast, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


# =============================================================================
# FOM evaluation for an arbitrary PSD
# =============================================================================

def fom(seq, freqs, Fe, sens_sq, S_func):
    kubo = float(simpson(Fe * S_func(freqs), x=freqs) / (2 * np.pi))
    if kubo > 0 and sens_sq > 0:
        return sens_sq / kubo
    return 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    # ── Pre-build GPS sequences and compute filter functions ──────────────────
    gps_ms = [1, 2, 4, 8, 16]
    gps_colors = {1: 'C0', 2: 'C4', 4: 'C5', 8: 'C2', 16: 'C6'}

    print('Building GPS sequences ...')
    gps_data = {}
    for m in gps_ms:
        seq, omega = build_gps(system, m)
        _, ss = detuning_sensitivity(seq)
        freqs, Fe, _, _ = fft_three_level_filter(
            seq, n_samples=N_FFT, pad_factor=PAD_FACTOR, m_y=1.0)
        gps_data[m] = dict(seq=seq, omega=omega, sens_sq=ss, freqs=freqs, Fe=Fe)
        print(f'  GPS m={m:2d}  Omega={omega:.4f}  sens={ss:.4f}')

    seq_R = build_ramsey(system)
    _, ss_R = detuning_sensitivity(seq_R)
    fR, FeR, _, _ = fft_three_level_filter(
        seq_R, n_samples=N_FFT, pad_factor=PAD_FACTOR, m_y=1.0)
    ramsey_data = dict(sens_sq=ss_R, freqs=fR, Fe=FeR)

    # ── Sweep: FOM vs omega_c2 for several omega_c1 values ───────────────────
    omega_c1_vals = [0.0, 0.5, 1.0, 4.0]
    omega_c2_sweep = np.linspace(1.0, 20.0, 200)

    fig1, axes = plt.subplots(
        2, 2, figsize=(12, 9), sharex=True,
        constrained_layout=True)
    axes = axes.ravel()

    for ax_idx, omega_c1 in enumerate(omega_c1_vals):
        ax = axes[ax_idx]
        # Ramsey
        foms_R = [fom(None, ramsey_data['freqs'], ramsey_data['Fe'],
                      ramsey_data['sens_sq'],
                      mixed_noise_psd(omega_c1, wc2))
                  for wc2 in omega_c2_sweep]
        ax.plot(omega_c2_sweep, foms_R, color='C3', lw=1.5, ls='--',
                label='Ramsey')
        # GPS lines
        for m in gps_ms:
            d = gps_data[m]
            foms_m = [fom(None, d['freqs'], d['Fe'], d['sens_sq'],
                          mixed_noise_psd(omega_c1, wc2))
                      for wc2 in omega_c2_sweep]
            ax.plot(omega_c2_sweep, foms_m,
                    color=gps_colors[m], lw=1.5,
                    label=rf'GPS $m={m}$')
        # Vertical lines at GPS Rabi frequencies
        for m in gps_ms:
            ax.axvline(gps_data[m]['omega'], color=gps_colors[m],
                       lw=0.5, ls=':', alpha=0.4)
        ax.set_title(
            rf'$\omega_{{c1}} = {omega_c1}$ rad s$^{{-1}}$', fontsize=11)
        ax.set_ylabel(r'FOM', fontsize=10)
        ax.set_ylim([0, None])
        ax.grid(True, alpha=0.3)
        if ax_idx >= 2:
            ax.set_xlabel(r'$\omega_{c2}$ (rad s$^{-1}$)', fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=7.5, ncol=2)

    fig1.suptitle(
        r'FOM vs high-frequency cutoff $\omega_{c2}$'
        '\n'
        r'Noise: $S(\omega) = (1 + \omega_{c1}/|\omega|)\,\theta(\omega_{c2}-|\omega|)$',
        fontsize=12)
    for ext in ['pdf', 'png']:
        fig1.savefig(OUTPUT_DIR / f'mixed_noise_fom_sweep.{ext}',
                     dpi=150, bbox_inches='tight')
    print('Saved mixed_noise_fom_sweep')

    # ── Point optimisations at representative (omega_c1, omega_c2) pairs ─────
    opt_cases = [
        # (omega_c1, omega_c2,  label)
        (0.0,  np.inf, 'Pure white'),
        (1.0,  np.inf, r'$\omega_{c1}=1$, no cutoff'),
        (1.0,  8.0,    r'$\omega_{c1}=1$, $\omega_{c2}=8$'),
        (1.0,  16.0,   r'$\omega_{c1}=1$, $\omega_{c2}=16$'),
        (4.0,  16.0,   r'$\omega_{c1}=4$, $\omega_{c2}=16$'),
        (4.0,  np.inf, r'$\omega_{c1}=4$, no cutoff'),
    ]

    print('\n--- Point optimisations (equiangular N=4) ---')
    hdr = (f'{"Noise case":<36} {"Omega*":>8} {"Omega*T":>8}'
           f' {"FOM_eq":>8} {"FOM_R":>7}'
           + ''.join(f' {"GPS"+str(m):>8}' for m in gps_ms))
    print(hdr)
    print('-' * len(hdr))

    opt_results = []   # (label, omega_opt, fom_eq, fom_R, {m: fom_m})

    for omega_c1, omega_c2, label in opt_cases:
        S_func = mixed_noise_psd(omega_c1, omega_c2)
        om_max = min(omega_c2 * 1.5, 40.0) if np.isfinite(omega_c2) else 40.0
        r = optimize_equiangular_sequence(
            system, T, N=4, noise_psd=S_func,
            omega_max=om_max,
            n_restarts=12, seed=42, n_fft=2048, pad_factor=4,
            popsize=20, maxiter=500,
        )
        _, ss_eq = detuning_sensitivity(r.seq)
        fq, Fe_eq, _, _ = fft_three_level_filter(
            r.seq, n_samples=N_FFT, pad_factor=PAD_FACTOR, m_y=1.0)
        fom_eq = fom(None, fq, Fe_eq, ss_eq, S_func)
        fom_R  = fom(None, ramsey_data['freqs'], ramsey_data['Fe'],
                     ramsey_data['sens_sq'], S_func)
        fom_gps = {m: fom(None, gps_data[m]['freqs'], gps_data[m]['Fe'],
                          gps_data[m]['sens_sq'], S_func)
                   for m in gps_ms}

        row = (f'{label:<36} {r.omega:>8.3f} {r.omega*T:>8.4f}'
               f' {fom_eq:>8.2f} {fom_R:>7.2f}'
               + ''.join(f' {fom_gps[m]:>8.2f}' for m in gps_ms))
        print(row)
        opt_results.append((label, r.omega, r.phases.copy(),
                            fom_eq, fom_R, fom_gps))

    # ── Figure 2: optimal Omega* vs omega_c2, for each omega_c1 ──────────────
    # Use a coarser grid for speed (evaluating only, not optimising)
    # Instead show which GPS m wins at each (omega_c1, omega_c2)
    omega_c2_grid = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0])

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax_idx, omega_c1 in enumerate([0.5, 2.0]):
        ax = axes2[ax_idx]
        # For each omega_c2, find GPS m with best FOM
        best_m_vals = []
        best_fom_vals = []
        for wc2 in omega_c2_grid:
            S_f = mixed_noise_psd(omega_c1, wc2)
            foms_m = {m: fom(None, gps_data[m]['freqs'], gps_data[m]['Fe'],
                              gps_data[m]['sens_sq'], S_f)
                      for m in gps_ms}
            best_m   = max(foms_m, key=foms_m.get)
            best_m_vals.append(best_m)
            best_fom_vals.append(foms_m[best_m])

        # Plot FOM curves for each GPS m
        for m in gps_ms:
            foms_m_arr = [fom(None, gps_data[m]['freqs'], gps_data[m]['Fe'],
                              gps_data[m]['sens_sq'],
                              mixed_noise_psd(omega_c1, wc2))
                          for wc2 in omega_c2_grid]
            ax.plot(omega_c2_grid, foms_m_arr, 'o-',
                    color=gps_colors[m], lw=1.5, ms=5,
                    label=rf'GPS $m={m}$')
        # Ramsey
        foms_R_arr = [fom(None, ramsey_data['freqs'], ramsey_data['Fe'],
                          ramsey_data['sens_sq'],
                          mixed_noise_psd(omega_c1, wc2))
                      for wc2 in omega_c2_grid]
        ax.plot(omega_c2_grid, foms_R_arr, 's--',
                color='C3', lw=1.5, ms=5, label='Ramsey')

        # Mark GPS Rabi harmonics
        for m in gps_ms:
            if gps_data[m]['omega'] <= 21:
                ax.axvline(gps_data[m]['omega'], color=gps_colors[m],
                           lw=0.5, ls=':', alpha=0.4)

        ax.set_title(rf'$\omega_{{c1}} = {omega_c1}$ rad s$^{{-1}}$', fontsize=11)
        ax.set_xlabel(r'$\omega_{c2}$ (rad s$^{-1}$)', fontsize=11)
        ax.set_ylabel(r'FOM', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig2.suptitle(
        r'GPS FOM vs $\omega_{c2}$: which $m$ wins?'
        '\n'
        r'$S(\omega) = (1 + \omega_{c1}/|\omega|)\,\theta(\omega_{c2}-|\omega|)$',
        fontsize=12)
    for ext in ['pdf', 'png']:
        fig2.savefig(OUTPUT_DIR / f'mixed_noise_gps_sweep.{ext}',
                     dpi=150, bbox_inches='tight')
    print('Saved mixed_noise_gps_sweep')

    plt.close('all')


if __name__ == '__main__':
    main()
