"""
Filter function comparison: Ramsey, GPS (integer cycle), and equiangular N=4,8,16.

All protocols share total interrogation time T = 2*pi.
All GPS protocols operate at delta=0 (resonant quadrature).

Protocol         Omega       delta     description
-------          -----       -----     -----------
Ramsey           20*pi       0         pi/2 - free - pi/2
GPS m=1          1           0         1 complete Rabi cycle
GPS m=8          8           0         8 complete Rabi cycles
Equiangular N=4  ~1.21       0         4 segments, optimised phases
Equiangular N=8  ~2.49       0         8 segments, optimised phases
Equiangular N=16 ~2.60       0         16 segments, optimised phases

NOTE on GPS sensitivity at delta=0 (H = delta*|e><e| convention):
    With H = delta*|e><e|, the derivative d<M>/ddelta at delta=0 is T/2 for
    ANY GPS m (since Omega_R=Omega at delta=0, and Omega*T=2*pi*m gives
    T*Omega*cos(m*pi)/2 = +-T/2).  So sens_sq = T^2/4 = pi^2 for all GPS m,
    identical to Ramsey.  The protocols differ only through their filter
    functions (and hence their Kubo variances).

Equiangular phases are loaded exclusively from the newest timestamped
equiangular_opt_cache_*.npz file (white-noise optimised).
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
    analytic_filter,
    detuning_sensitivity,
)
from quantum_pulse_suite.analysis.pulse_optimizer import (
    white_noise_psd,
    one_over_f_psd,
)

CACHE_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'
CACHE_PREFIX = 'equiangular_opt_cache'


# =============================================================================
# Parameters
# =============================================================================

T          = 2 * np.pi       # total interrogation time
OMEGA_FAST = 20 * np.pi      # Rabi frequency for fast pi/2 pulses in Ramsey

N_FFT      = 4096          # time samples for direct DFT
OMEGA_PLOT = np.logspace(-2, np.log10(30), 800)   # log-spaced for smooth loglog

# GPS m=1: Omega = 2*pi*1/T = 1 rad/s (1 complete cycle in time T)
OMEGA_GPS1 = 2 * np.pi * 1 / T    # = 1.0

# GPS m=8: Omega = 2*pi*8/T = 8 rad/s (8 complete cycles in time T)
OMEGA_GPS8 = 2 * np.pi * 8 / T    # = 8.0

# Both GPS protocols at resonant quadrature (delta=0).
# At delta=0 with Omega*T = 2*pi*m, sens_sq = pi^2 for any m.
DELTA_GPS1 = 0.0
DELTA_GPS8 = 0.0

S_white = white_noise_psd()
S_1_over_f = one_over_f_psd()

OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'


def find_latest_equiangular_cache():
    """Return the newest timestamped equiangular cache, or a legacy path."""
    matches = sorted(CACHE_DIR.glob(f'{CACHE_PREFIX}_*.npz'))
    if matches:
        return matches[-1]
    legacy = CACHE_DIR / 'equiangular_opt_cache.npz'
    if legacy.exists():
        return legacy
    return None


def load_cached_equiangular_result(system, cache, N):
    """Rebuild cached white-noise equiangular result for a given N."""
    omega = float(cache[f'eq_N{N}_white_omega'])
    phases = np.asarray(cache[f'eq_N{N}_white_phases'], dtype=float)
    seq = build_equiangular(system, omega, phases)
    return omega, phases, seq


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

OMEGA_MIN = 2.0 * np.pi / T   # Fourier limit: lowest resolvable frequency in time T


def evaluate(seq, label=''):
    """Return (sens_sq, noise_white, sigma_nu_white, noise_1f, sigma_nu_1f, freqs, Fe).

    Filter function F(w) = |Chi(w)|^2 on a log-spaced grid for smooth log-log
    plots.  Noise integrals are bandlimited from OMEGA_MIN = 2*pi/T to infinity
    (noise slower than one measurement cycle cannot be resolved).
    sigma_nu = noise_var / F(0)  where  F(0) = sens_sq  by DC consistency.
    """
    _, sens_sq = detuning_sensitivity(seq)

    # Smooth plot grid: log-spaced, evaluated via analytic Fourier integrals
    freqs, Fe = analytic_filter(seq, OMEGA_PLOT, m_y=1.0)

    # Noise integrals: use FFT on a dense grid for accurate integration
    # Integrate from OMEGA_MIN = 2*pi/T (Fourier limit)
    freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
        seq, n_samples=N_FFT, pad_factor=4, m_y=1.0)
    mask     = freqs_fft >= OMEGA_MIN
    noise_w  = float(simpson(Fe_fft[mask] * S_white(freqs_fft[mask]),    x=freqs_fft[mask]) / (2 * np.pi))
    noise_f  = float(simpson(Fe_fft[mask] * S_1_over_f(freqs_fft[mask]), x=freqs_fft[mask]) / (2 * np.pi))
    snu_w    = noise_w / sens_sq  if (sens_sq > 0 and noise_w  > 0) else 0.0
    snu_f    = noise_f / sens_sq  if (sens_sq > 0 and noise_f  > 0) else 0.0
    if label:
        print(f'  {label:<28}  sens={sens_sq:.4f}  noise_w={noise_w:.3e}  sigma_nu_w={snu_w:.3e}')
    return sens_sq, noise_w, snu_w, noise_f, snu_f, freqs, Fe


def objective_score(sens_sq, noise_var, sens_ref, noise_ref, weight=1.0):
    """Ramsey-normalized post-hoc optimizer score."""
    if sens_ref <= 0.0 or noise_ref <= 0.0:
        return 0.0
    return sens_sq / sens_ref - weight * noise_var / noise_ref


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    # ── Build sequences ───────────────────────────────────────────────────────
    print('Building Ramsey (pi/2 - free - pi/2) ...')
    seq_ramsey = build_ramsey(system)

    print('Building GPS m=1 (delta=0, resonant quadrature) ...')
    seq_gps1 = build_gps_at_zero(system, OMEGA_GPS1, DELTA_GPS1)

    print('Building GPS m=8 (delta=0, resonant quadrature) ...')
    seq_gps8 = build_gps_at_zero(system, OMEGA_GPS8, DELTA_GPS8)

    cache_path = find_latest_equiangular_cache()
    if cache_path is None:
        raise FileNotFoundError(
            f'No equiangular cache found matching {CACHE_PREFIX}_*.npz in {CACHE_DIR}. '
            'Run scripts/run_equiangular_optimization.py first.'
        )
    _cache = np.load(str(cache_path), allow_pickle=True)
    print(f'Loading equiangular N=4, N=8, N=16 from cache: {cache_path}')

    omega_eq4, phases_eq4, seq_eq4 = load_cached_equiangular_result(system, _cache, 4)
    print(f'  N=4:  Omega*T = {omega_eq4 * T:.5f}')

    omega_eq8, phases_eq8, seq_eq8 = load_cached_equiangular_result(system, _cache, 8)
    print(f'  N=8:  Omega*T = {omega_eq8 * T:.5f}')

    omega_eq16, phases_eq16, seq_eq16 = load_cached_equiangular_result(system, _cache, 16)
    print(f'  N=16: Omega*T = {omega_eq16 * T:.5f}')

    # ── Evaluate all protocols ────────────────────────────────────────────────
    print('\nEvaluating ...')
    res_R   = evaluate(seq_ramsey, 'Ramsey')
    res_G1  = evaluate(seq_gps1,  'GPS m=1 (d=0, resonant)')
    res_G8  = evaluate(seq_gps8,  'GPS m=8 (d=0, resonant)')
    res_E4  = evaluate(seq_eq4,   f'Equiangular N=4 (OmegaT={omega_eq4*T:.3f})')
    res_E8  = evaluate(seq_eq8,   f'Equiangular N=8 (OmegaT={omega_eq8*T:.3f})')
    res_E16 = evaluate(seq_eq16,  f'Equiangular N=16 (OmegaT={omega_eq16*T:.3f})')

    # Print table
    print()
    hdr = f'{"Protocol":<36} {"sens_sq":>9} {"noise_w":>11} {"snu_w":>11} {"snu_1/f":>11}'
    print(hdr)
    print('-' * len(hdr))
    rows = [
        ('Ramsey',                *res_R[:5]),
        ('GPS m=1 (d=0)',         *res_G1[:5]),
        ('GPS m=8 (d=0)',         *res_G8[:5]),
        ('Equiangular N=4',       *res_E4[:5]),
        ('Equiangular N=8',       *res_E8[:5]),
        ('Equiangular N=16',      *res_E16[:5]),
    ]
    sens_ref = res_R[0]
    noise_w_ref = res_R[1]
    noise_f_ref = res_R[3]
    for lbl, s, nw, sw, nf, sf in rows:
        print(f'{lbl:<36} {s:>9.4f} {nw:>11.3e} {sw:>11.3e} {sf:>11.3e}')

    print()
    hdr2 = f'{"Protocol":<36} {"score_w":>11} {"score_1/f":>11}'
    print(hdr2)
    print('-' * len(hdr2))
    for lbl, s, nw, sw, nf, sf in rows:
        sc_w = objective_score(s, nw, sens_ref, noise_w_ref)
        sc_f = objective_score(s, nf, sens_ref, noise_f_ref)
        print(f'{lbl:<36} {sc_w:>11.3f} {sc_f:>11.3f}')

    # ── Filter function plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    f_R,   Fe_R   = res_R[5],   res_R[6]
    f_G1,  Fe_G1  = res_G1[5],  res_G1[6]
    f_G8,  Fe_G8  = res_G8[5],  res_G8[6]
    f_E4,  Fe_E4  = res_E4[5],  res_E4[6]
    f_E8,  Fe_E8  = res_E8[5],  res_E8[6]
    f_E16, Fe_E16 = res_E16[5], res_E16[6]

    ax.loglog(f_R,   Fe_R   / T**2 + 1e-20, color='C3', lw=2,   label=r'Ramsey')
    ax.loglog(f_G1,  Fe_G1  / T**2 + 1e-20, color='C0', lw=2,   label=r'GPS $m{=}1$')
    ax.loglog(f_G8,  Fe_G8  / T**2 + 1e-20, color='C2', lw=2,   label=r'GPS $m{=}8$')
    ax.loglog(f_E4,  Fe_E4  / T**2 + 1e-20, color='C1', lw=2.5, label=r'Equiangular $N{=}4$')
    ax.loglog(f_E8,  Fe_E8  / T**2 + 1e-20, color='C4', lw=2.5, label=r'Equiangular $N{=}8$')
    ax.loglog(f_E16, Fe_E16 / T**2 + 1e-20, color='C5', lw=2.5, label=r'Equiangular $N{=}16$')

    # Vertical dashed lines at Omega_GPS1 and Omega_GPS8 harmonics
    for k in range(1, 10):
        if OMEGA_GPS1 * k <= 30:
            ax.axvline(OMEGA_GPS1 * k, color='C0', lw=0.5, ls=':', alpha=0.4)
    for k in range(1, 5):
        if OMEGA_GPS8 * k <= 30:
            ax.axvline(OMEGA_GPS8 * k, color='C2', lw=0.5, ls=':', alpha=0.4)

    ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=12)
    ax.set_ylabel(r'$F(\omega)\,/\,T^2$', fontsize=12)
    ax.set_title(
        r'Three-level clock filter functions at equal total time $T = 2\pi$'
        '\n'
        r'All protocols at $\delta = 0$ ($H_\delta = \delta|e\rangle\langle e|$)',
        fontsize=11)
    ax.set_xlim([3e-2, 30])
    ax.set_ylim([1e-9, 1.5])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8.5, loc='lower left')
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'protocol_comparison.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')

    # Return key results for reference
    return {
        'omega_eq4':  omega_eq4,
        'phases_eq4': phases_eq4,
        'rows':       rows,
        'score_rows': [
            (lbl,
             objective_score(s, nw, sens_ref, noise_w_ref),
             objective_score(s, nf, sens_ref, noise_f_ref))
            for lbl, s, nw, sw, nf, sf in rows
        ],
    }


if __name__ == '__main__':
    main()
