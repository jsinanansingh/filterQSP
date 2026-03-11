"""
Pulse-shaped GPS filter functions and sigma_nu comparison.

All shaped protocols satisfy the boundary conditions:
  (i)  Omega(0) = Omega(T) = 0  (smooth on/off)
  (ii) integral_0^T Omega(t) dt = 2*pi*m  (integer Rabi cycles -> same
       final state as square GPS, sensitivity preserved)

Shapes compared (for m=1 and m=8):
  Square      -- constant Omega (standard GPS, reference)
  Raised cosine (Hann) -- Omega(t) = Omega_mean*(1 - cos(2*pi*t/T))
  Blackman    -- Omega(t) proportional to 0.42 - 0.5*cos(2pi*t/T) + 0.08*cos(4pi*t/T),
                 normalised so the total rotation = 2*pi*m.

Filter functions are computed on a dense discretised sequence (N_DISC segments)
via analytic_filter, giving exact results for piecewise-constant approximations.
Detuning sensitivities use the same discretised sequences.
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
    analytic_filter,
    detuning_sensitivity,
    raised_cosine_filter,
    default_omega_cutoff,
)
from quantum_pulse_suite.analysis.pulse_optimizer import white_noise_psd, one_over_f_psd

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi
N_DISC     = 512           # discretisation steps for shaped envelopes
FREQS      = np.logspace(-1.5, np.log10(30), 600)
OMEGA_CUTOFF = default_omega_cutoff(T)
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'


# =============================================================================
# Envelope definitions  (each returns array of shape (N,) given t midpoints)
# All normalised so  (1/T) * integral Omega(t) dt = Omega_mean
# =============================================================================

def envelope_square(ts, T, omega_mean):
    return np.full_like(ts, omega_mean)


def envelope_hann(ts, T, omega_mean):
    """Raised-cosine (Hann): Omega(0)=Omega(T)=0, peak = 2*omega_mean."""
    return omega_mean * (1.0 - np.cos(2.0 * np.pi * ts / T))


def envelope_blackman(ts, T, omega_mean):
    """Blackman window normalised to mean = omega_mean. Omega(0)=Omega(T)=0."""
    # Standard Blackman: a0 - a1*cos(2pi*t/T) + a2*cos(4pi*t/T)
    # Integral over [0,T] = a0*T  ->  scale so mean = omega_mean
    a0, a1, a2 = 0.42, 0.50, 0.08
    return (omega_mean / a0) * (a0
                                - a1 * np.cos(2.0 * np.pi * ts / T)
                                + a2 * np.cos(4.0 * np.pi * ts / T))


# =============================================================================
# Build discretised sequence from an envelope function
# =============================================================================

def build_shaped_seq(system, omega_mean, m_cycles, envelope_fn, n_disc=N_DISC):
    """
    Discretise a shaped GPS sequence into n_disc equal-time segments.
    The drive axis is x (phi=0) throughout.
    """
    T_total = T
    tau     = T_total / n_disc
    ts_mid  = (np.arange(n_disc) + 0.5) * tau   # segment midpoints
    omegas  = envelope_fn(ts_mid, T_total, omega_mean)

    # Enforce non-negativity (Blackman can dip slightly negative at edges)
    omegas  = np.maximum(omegas, 0.0)

    seq = MultiLevelPulseSequence(system, system.probe)
    for omega_k in omegas:
        seq.add_continuous_pulse(omega_k, [1, 0, 0], 0.0, tau)
    seq.compute_polynomials()
    return seq


def build_square_seq(system, omega_mean):
    """Single continuous x-axis segment (standard GPS)."""
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(omega_mean, [1, 0, 0], 0.0, T)
    seq.compute_polynomials()
    return seq


def build_ramsey(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq


# =============================================================================
# Evaluate: filter function + sigma_nu
# =============================================================================

S_white    = white_noise_psd()
S_1_over_f = one_over_f_psd()


def evaluate_seq(seq, label=''):
    _, sens_sq = detuning_sensitivity(seq)
    _, Fe      = analytic_filter(seq, FREQS, m_y=1.0)

    # Dense FFT grid for accurate Kubo integrals
    from quantum_pulse_suite.core.three_level_filter import fft_three_level_filter
    freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
        seq, n_samples=4096, pad_factor=4, m_y=1.0)
    mask = freqs_fft >= OMEGA_CUTOFF
    kubo_w = float(simpson(Fe_fft[mask] * S_white(freqs_fft[mask]),    x=freqs_fft[mask]) / (2*np.pi))
    kubo_f = float(simpson(Fe_fft[mask] * S_1_over_f(freqs_fft[mask]), x=freqs_fft[mask]) / (2*np.pi))
    sigma_nu_w = kubo_w / sens_sq if (sens_sq > 0 and kubo_w > 0) else 0.0
    sigma_nu_f = kubo_f / sens_sq if (sens_sq > 0 and kubo_f > 0) else 0.0

    if label:
        print(f'  {label:<36}  sens={sens_sq:.4f}  kubo_w={kubo_w:.3e}'
              f'  sigma_nu_w={sigma_nu_w:.4f}  sigma_nu_1/f={sigma_nu_f:.4f}')
    return dict(sens_sq=sens_sq, kubo_w=kubo_w, sigma_nu_w=sigma_nu_w,
                kubo_f=kubo_f, sigma_nu_f=sigma_nu_f, Fe=Fe)


# For the square GPS, use raised_cosine_filter to get the Hann filter function
# analytically via the existing implementation.  For square, just analytic_filter.
def evaluate_square_via_rc(seq, label=''):
    """Evaluate square GPS but also compute Hann Fe via raised_cosine_filter."""
    _, Fe_sq = analytic_filter(seq, FREQS, m_y=1.0)
    _, Fe_rc = raised_cosine_filter(seq, FREQS, m_y=1.0)
    return Fe_sq, Fe_rc


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    omega_gps1 = 2 * np.pi * 1 / T   # = 1.0  rad/s
    omega_gps8 = 2 * np.pi * 8 / T   # = 8.0  rad/s

    shapes = [
        ('Square',       envelope_square,   '-',  None),
        ('Hann (RC)',     envelope_hann,     '--', None),
        ('Blackman',     envelope_blackman,  ':',  None),
    ]

    print('Building sequences ...')
    seqs = {}
    for m_cyc, omega in [(1, omega_gps1), (8, omega_gps8)]:
        seqs[m_cyc] = {}
        for name, fn, _, _ in shapes:
            seqs[m_cyc][name] = build_shaped_seq(system, omega, m_cyc, fn)

    seq_ramsey = build_ramsey(system)

    print('\nEvaluating ...')
    results = {}
    results['Ramsey'] = evaluate_seq(seq_ramsey, 'Ramsey')

    for m_cyc in [1, 8]:
        results[m_cyc] = {}
        for name, _, _, _ in shapes:
            lbl = f'GPS m={m_cyc}  {name}'
            results[m_cyc][name] = evaluate_seq(seqs[m_cyc][name], lbl)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    hdr = f'{"Protocol":<38} {"sigma_nu w":>12} {"sigma_nu 1/f":>14} {"sens_sq":>9}'
    print(hdr)
    print('-' * len(hdr))
    print(f'  {"Ramsey":<36} {results["Ramsey"]["sigma_nu_w"]:>12.4f}'
          f' {results["Ramsey"]["sigma_nu_f"]:>14.4f}'
          f' {results["Ramsey"]["sens_sq"]:>9.4f}')
    for m_cyc in [1, 8]:
        for name, _, _, _ in shapes:
            r = results[m_cyc][name]
            lbl = f'GPS m={m_cyc}  {name}'
            print(f'  {lbl:<36} {r["sigma_nu_w"]:>12.4f} {r["sigma_nu_f"]:>14.4f}'
                  f' {r["sens_sq"]:>9.4f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    colors_m1 = ['C0', 'C9', 'C1']   # blue family for m=1
    colors_m8 = ['C2', 'C3', 'C4']   # green/red family for m=8
    ls_map    = {name: ls for name, _, ls, _ in shapes}

    for ax, m_cyc, colors, title in [
        (axes[0], 1, colors_m1, r'GPS $m{=}1$  ($\Omega_{\rm mean}{=}1$)'),
        (axes[1], 8, colors_m8, r'GPS $m{=}8$  ($\Omega_{\rm mean}{=}8$)'),
    ]:
        # Ramsey as grey reference
        ax.loglog(FREQS, results['Ramsey']['Fe'] / T**2 + 1e-20,
                  color='0.55', lw=1.5, ls='-', label='Ramsey')

        for (name, _, ls, _), col in zip(shapes, colors):
            r   = results[m_cyc][name]
            lbl = rf'{name}  ($\sigma_{{\nu,w}}$={r["sigma_nu_w"]:.3f})'
            ax.loglog(FREQS, r['Fe'] / T**2 + 1e-20,
                      color=col, lw=2, ls=ls, label=lbl)

        # Harmonic markers for m=8
        omega_mean = 2 * np.pi * m_cyc / T
        if m_cyc == 8:
            for k in range(1, 5):
                if omega_mean * k <= 30:
                    ax.axvline(omega_mean * k, color='0.7', lw=0.6, ls=':', alpha=0.5)

        ax.set_xlim([FREQS[0], 30])
        ax.set_ylim([1e-9, 1.5])
        ax.set_xlabel(r'$\omega$ (rad s$^{-1}$)', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8.5, loc='lower left')

    axes[0].set_ylabel(r'$F_e(\omega)\,/\,T^2$', fontsize=11)
    fig.suptitle(
        r'Pulse-shaped GPS: $F_e(\omega)$ at $T{=}2\pi$, $\delta{=}0$'
        '\n'
        r'All shapes satisfy $\int_0^T\!\Omega(t)\,dt = 2\pi m$  '
        r'and  $\Omega(0){=}\Omega(T){=}0$ (Hann, Blackman)',
        fontsize=10)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'shaped_gps_filter_functions.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')


if __name__ == '__main__':
    main()
