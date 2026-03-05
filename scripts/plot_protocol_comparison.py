"""
Filter function comparison: Ramsey, GPS (integer cycle), and optimal equiangular N=4.

All protocols share total interrogation time T = 2*pi.

Protocol        Omega       delta (operating point)     description
-------         -----       -----------------------     -----------
Ramsey          20*pi       0                           pi/2 - free - pi/2
GPS m=1         1           sqrt(15)  (k=4 zero)        1 complete Rabi cycle
GPS m=8         8           sqrt(17)  (k=9 zero)        8 complete Rabi cycles
Equiangular N=4 ~ pi/10T    0                           4 segments, optimised phases

GPS zero crossings:
    Signal: S(delta) = -(delta/Omega_eff) * sin(m*pi*Omega_eff/Omega)
            Omega_eff = sqrt(Omega^2 + delta^2)
    Zeros:  delta_k = Omega * sqrt((k/m)^2 - 1),  k >= m+1 integer
    Slope:  dS/ddelta|_{delta_k} = -(-1)^k * (m*pi/Omega) * (1 - m^2/k^2)

    GPS m=1, k=4:  delta = sqrt(15) * Omega,  sens_sq = (pi * 15/16)^2
    GPS m=8, k=9:  delta = sqrt(17/64) * Omega = sqrt(17)/8 * Omega,
                   sens_sq = (pi * 17/81)^2
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
from quantum_pulse_suite.analysis.pulse_optimizer import (
    optimize_equiangular_sequence,
    white_noise_psd,
    one_over_f_psd,
)


# =============================================================================
# Parameters
# =============================================================================

T          = 2 * np.pi       # total interrogation time
OMEGA_FAST = 20 * np.pi      # Rabi frequency for fast pi/2 pulses in Ramsey

N_FFT      = 4096
PAD_FACTOR = 4

# GPS m=1: Omega = 2*pi*1/T = 1 rad/s (1 complete cycle in time T)
OMEGA_GPS1 = 2 * np.pi * 1 / T    # = 1.0

# GPS m=8: Omega = 2*pi*8/T = 8 rad/s (8 complete cycles in time T)
OMEGA_GPS8 = 2 * np.pi * 8 / T    # = 8.0

# Zero-crossing operating points: delta_k = Omega * sqrt((k/m)^2 - 1)
# GPS m=1, k=4:  delta = 1 * sqrt(16 - 1) = sqrt(15) ≈ 3.873
DELTA_GPS1_K4 = OMEGA_GPS1 * np.sqrt((4 / 1.0) ** 2 - 1)   # sqrt(15)

# GPS m=8, k=9:  delta = 8 * sqrt((9/8)^2 - 1) = 8 * sqrt(17/64) = sqrt(17) ≈ 4.123
DELTA_GPS8_K9 = OMEGA_GPS8 * np.sqrt((9 / 8.0) ** 2 - 1)   # sqrt(17)

S_white = white_noise_psd()
S_1_over_f = one_over_f_psd()

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'


# =============================================================================
# Sequence builders
# =============================================================================

def build_ramsey(system):
    """pi/2 - free_evolution - pi/2 with fast Rabi pulses."""
    tau_pi2  = np.pi / (2 * OMEGA_FAST)   # duration of each pi/2 pulse
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


def build_gps_at_zero(system, omega, delta_zero):
    """
    GPS: single continuous Rabi drive for full time T at detuning delta_zero.

    Building with the operating-point detuning baked in means:
    - fft_three_level_filter computes Fe at that operating point
    - detuning_sensitivity returns the slope at that operating point
    """
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega, [1, 0, 0], delta_zero, T)
    seq.compute_polynomials()
    return seq


def build_equiangular(system, omega, phases):
    """N-segment equiangular sequence with given phases, delta=0."""
    N   = len(phases)
    tau = T / N
    seq = MultiLevelPulseSequence(system, system.probe)
    for phi in phases:
        seq.add_continuous_pulse(omega, [np.cos(phi), np.sin(phi), 0.0], 0.0, tau)
    seq.compute_polynomials()
    return seq


# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate(seq, label=''):
    """Return (sens_sq, kubo_white, fom_white, kubo_1f, fom_1f, freqs, Fe)."""
    _, sens_sq = detuning_sensitivity(seq)
    freqs, Fe, _, _ = fft_three_level_filter(
        seq, n_samples=N_FFT, pad_factor=PAD_FACTOR, m_y=1.0)
    kubo_w  = float(simpson(Fe * S_white(freqs),   x=freqs) / (2 * np.pi))
    kubo_f  = float(simpson(Fe * S_1_over_f(freqs), x=freqs) / (2 * np.pi))
    fom_w   = sens_sq / kubo_w  if (sens_sq > 0 and kubo_w  > 0) else 0.0
    fom_f   = sens_sq / kubo_f  if (sens_sq > 0 and kubo_f  > 0) else 0.0
    if label:
        print(f'  {label:<28}  sens={sens_sq:.4f}  kubo_w={kubo_w:.3e}  FOM_w={fom_w:.1f}')
    return sens_sq, kubo_w, fom_w, kubo_f, fom_f, freqs, Fe


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    # ── Build sequences ───────────────────────────────────────────────────────
    print('Building Ramsey (pi/2 - free - pi/2) ...')
    seq_ramsey = build_ramsey(system)

    print(f'Building GPS m=1, k=4 zero (delta={DELTA_GPS1_K4:.4f}) ...')
    seq_gps1 = build_gps_at_zero(system, OMEGA_GPS1, DELTA_GPS1_K4)

    print(f'Building GPS m=8, k=9 zero (delta={DELTA_GPS8_K9:.4f}) ...')
    seq_gps8 = build_gps_at_zero(system, OMEGA_GPS8, DELTA_GPS8_K9)

    print('Optimising equiangular N=4 (white noise) ...')
    result_eq4 = optimize_equiangular_sequence(
        system, T, N=4, noise_psd='white',
        n_restarts=8, seed=42, n_fft=2048, pad_factor=4,
        popsize=15, maxiter=400,
    )
    omega_eq4  = result_eq4.omega
    phases_eq4 = result_eq4.phases
    seq_eq4    = result_eq4.seq
    print(f'  Omega = {omega_eq4:.6f}  (Omega*T = {omega_eq4 * T:.5f})')
    print(f'  phases = {np.array2string(phases_eq4, precision=4, separator=", ")}')

    # ── Evaluate all protocols ────────────────────────────────────────────────
    print('\nEvaluating ...')
    res_R  = evaluate(seq_ramsey, 'Ramsey')
    res_G1 = evaluate(seq_gps1,  f'GPS m=1 k=4 (d={DELTA_GPS1_K4:.3f})')
    res_G8 = evaluate(seq_gps8,  f'GPS m=8 k=9 (d={DELTA_GPS8_K9:.3f})')
    res_E4 = evaluate(seq_eq4,   f'Equiangular N=4 (Omega={omega_eq4:.4f})')

    # Print table
    print()
    hdr = f'{"Protocol":<32} {"sens_sq":>9} {"kubo_w":>11} {"FOM_w":>11} {"FOM_1/f":>11}'
    print(hdr)
    print('-' * len(hdr))
    rows = [
        ('Ramsey',               *res_R[:5]),
        ('GPS m=1, k=4 zero',    *res_G1[:5]),
        ('GPS m=8, k=9 zero',    *res_G8[:5]),
        ('Equiangular N=4',      *res_E4[:5]),
    ]
    for lbl, s, kw, fw, kf, ff in rows:
        print(f'{lbl:<32} {s:>9.4f} {kw:>11.3e} {fw:>11.1f} {ff:>11.1f}')

    # ── Filter function plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    f_R,  Fe_R  = res_R[5],  res_R[6]
    f_G1, Fe_G1 = res_G1[5], res_G1[6]
    f_G8, Fe_G8 = res_G8[5], res_G8[6]
    f_E4, Fe_E4 = res_E4[5], res_E4[6]

    label_eq = (
        rf'Equiangular $N{{=}}4$, $\Omega T={omega_eq4 * T:.3f}$'
        rf', $\phi=[0,\,{phases_eq4[1]:.2f},\,{phases_eq4[2]:.2f},\,{phases_eq4[3]:.2f}]$'
    )

    ax.loglog(f_R,  Fe_R  / T**2 + 1e-20, color='C3', lw=2,   label=r'Ramsey')
    ax.loglog(f_G1, Fe_G1 / T**2 + 1e-20, color='C0', lw=2,
              label=r'GPS $m{=}1$, $k{=}4$ zero  ($\delta/\Omega = \sqrt{15}$)')
    ax.loglog(f_G8, Fe_G8 / T**2 + 1e-20, color='C2', lw=2,
              label=r'GPS $m{=}8$, $k{=}9$ zero  ($\delta/\Omega = \sqrt{17}/8$)')
    ax.loglog(f_E4, Fe_E4 / T**2 + 1e-20, color='C1', lw=2.5, label=label_eq)

    # Vertical dashed lines at Omega_GPS1 and Omega_GPS8 harmonics
    for k in range(1, 10):
        if OMEGA_GPS1 * k <= 30:
            ax.axvline(OMEGA_GPS1 * k, color='C0', lw=0.5, ls=':', alpha=0.4)
    for k in range(1, 5):
        if OMEGA_GPS8 * k <= 30:
            ax.axvline(OMEGA_GPS8 * k, color='C2', lw=0.5, ls=':', alpha=0.4)

    ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=12)
    ax.set_ylabel(r'$F_e(\omega)\,/\,T^2$', fontsize=12)
    ax.set_title(
        r'Three-level clock filter functions at equal total time $T = 2\pi$'
        '\n'
        r'GPS protocols at zero-crossing operating points; '
        r'equiangular at $\delta = 0$',
        fontsize=11)
    ax.set_xlim([3e-2, 30])
    ax.set_ylim([1e-9, 1.5])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8.5, loc='lower left')
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'protocol_comparison.{ext}'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')

    # Return key results for reference
    return {
        'omega_eq4':  omega_eq4,
        'phases_eq4': phases_eq4,
        'rows':       rows,
    }


if __name__ == '__main__':
    main()
