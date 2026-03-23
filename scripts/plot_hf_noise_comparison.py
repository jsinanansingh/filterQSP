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

Equiangular N=4/8/16 and QSP n=4/8/13 are loaded from the latest caches
(white-noise optimised sequences, same as protocol_comparison.py).
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
)
from quantum_pulse_suite.analysis.pulse_optimizer import build_qsp_3level

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.size':          8,
    'axes.labelsize':     9,
    'axes.titlesize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    6.5,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '0.7',
    'lines.linewidth':    1.4,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.minor.width':  0.4,
    'ytick.minor.width':  0.4,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
})

T          = 2 * np.pi
OMEGA_FAST = 20 * np.pi
OMEGA_GPS1 = 2 * np.pi * 1 / T   # = 1.0
OMEGA_GPS8 = 2 * np.pi * 8 / T   # = 8.0

EQ_NS  = [4, 8, 16]
QSP_NS = [4, 8, 13]
EQ_COLORS  = ['#E07020', '#C04000', '#802000']
QSP_COLORS = ['#2090D0', '#1060A0', '#083070']

def _sanitize(label):
    return label.replace('/', 'f').replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')

# Analytic frequency grid: fine enough for accurate Kubo integrals
# Use log-spacing from near-DC to capture the full filter function
OMEGA_ANA_MIN = 1e-4
OMEGA_ANA_MAX = 60.0
N_ANA         = 8000

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


def build_double_pi_qsp(system):
    """QSP n=2: theta=[pi,pi], phi=[0,0] — maximises DC sensitivity."""
    return build_qsp_3level(system, T, 2, [np.pi, np.pi], [0.0, 0.0], OMEGA_FAST)


# =============================================================================
# sigma_nu helpers  (lower is better)
# =============================================================================

def sigma_nu_highpass(freqs, Fe, sens_sq, cutoff):
    """sigma_nu with S(ω) = 1 for ω ≥ cutoff, else 0."""
    mask = freqs >= cutoff
    if mask.sum() < 2:
        return 0.0
    noise_var = float(simpson(Fe[mask], x=freqs[mask]) / (2 * np.pi))
    return noise_var / sens_sq if sens_sq > 0 else np.inf


def sigma_nu_lorentzian(freqs, Fe, sens_sq, omega_peak, width=0.5):
    """sigma_nu with a Lorentzian noise peak: S(ω) = 1/((ω-ω_peak)²+width²)."""
    S = 1.0 / ((freqs - omega_peak)**2 + width**2)
    noise_var = float(simpson(Fe * S, x=freqs) / (2 * np.pi))
    return noise_var / sens_sq if sens_sq > 0 else np.inf


# =============================================================================
# Main
# =============================================================================

# (eq_cache_key, qsp_noise_label, file_suffix, display_label)
NOISE_VARIANTS = [
    ('highpass2', 'High-pass (w_c=2)', '',    'high-pass'),
    ('1f',        '1/f',              '_1f',  r'$1/f$'),
]


def _build_protocols(system, seq_R, seq_G1, seq_G8, seq_dpi,
                     eq_cache, qsp_cache, eq_key, qsp_label):
    protocols = [
        ('Ramsey',                    seq_R,   'C3',      '-',  2.0),
        (r'QSP $n{=}2$ ($2\pi$-R)',  seq_dpi, '#9B30FF', ':',  2.0),
        ('GPS $m{=}1$',               seq_G1,  'C0',      '-',  2.0),
        ('GPS $m{=}8$',               seq_G8,  'C2',      '-',  2.0),
    ]
    for N, color in zip(EQ_NS, EQ_COLORS):
        omega  = float(eq_cache[f'eq_N{N}_{eq_key}_omega'])
        phases = np.asarray(eq_cache[f'eq_N{N}_{eq_key}_phases'], dtype=float)
        seq    = build_equiangular(system, omega, phases)
        protocols.append((rf'Eq $N{{={N}}}$', seq, color, '-', 1.8))
    if qsp_cache is not None:
        for n, color in zip(QSP_NS, QSP_COLORS):
            k = f'qsp_n{n}_{_sanitize(qsp_label)}'
            if f'{k}_thetas' not in qsp_cache:
                print(f'  WARNING: {k}_thetas missing, skipping n={n}')
                continue
            seq = build_qsp_3level(
                system, T, n,
                qsp_cache[f'{k}_thetas'], qsp_cache[f'{k}_phis'],
                float(qsp_cache[f'{k}_omega_fast']))
            protocols.append((rf'QSP $n{{={n}}}$', seq, color, '--', 1.8))
    return protocols


def _make_figures(protocols, data, file_suffix, opt_label):
    cutoffs     = np.linspace(0.0, 28.0, 500)
    omega_peaks = np.linspace(0.1, 28.0, 500)
    width       = 0.3

    # Figure 1: high-pass cutoff sweep
    fig1, ax1 = plt.subplots(figsize=(3.375, 2.8))
    for label, _, color, ls, lw in protocols:
        d    = data[label]
        snus = np.array([sigma_nu_highpass(d['freqs'], d['Fe'], d['sens_sq'], wc)
                         for wc in cutoffs])
        ax1.semilogy(cutoffs, np.clip(snus, 1e-8, None),
                     color=color, lw=lw, ls=ls, label=label)
    for k in range(1, 5):
        wk = OMEGA_GPS8 * k
        if wk <= 28:
            ax1.axvline(wk, color='C2', lw=0.8, ls=':', alpha=0.5)
            ax1.text(wk + 0.15, 1e-6, f'$8\\cdot{k}$', color='C2', fontsize=6, va='bottom')
    ax1.set_xlabel(r'Noise cutoff $\omega_c$ (rad s$^{-1}$)')
    ax1.set_ylabel(r'$\sigma^2_\nu(\omega_c)$')
    ax1.set_title(
        r'$\sigma^2_\nu$ under high-pass noise $S(\omega)=\theta(\omega-\omega_c)$'
        f'\n({opt_label}-noise opt., lower is better)')
    ax1.set_xlim([0, 28])
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(which='both', top=True, right=True)
    ax1.legend()
    fig1.tight_layout()
    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'hf_noise_fom_vs_cutoff{file_suffix}.{ext}'
        fig1.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    # Figure 2: Lorentzian peak sweep
    fig2, ax2 = plt.subplots(figsize=(3.375, 2.8))
    for label, _, color, ls, lw in protocols:
        d    = data[label]
        snus = np.array([sigma_nu_lorentzian(d['freqs'], d['Fe'], d['sens_sq'], wp, width=width)
                         for wp in omega_peaks])
        ax2.semilogy(omega_peaks, np.clip(snus, 1e-8, None),
                     color=color, lw=lw, ls=ls, label=label)
    for k in range(1, 5):
        wk = OMEGA_GPS8 * k
        if wk <= 28:
            ax2.axvline(wk, color='C2', lw=0.8, ls=':', alpha=0.5)
    for k in range(1, 10):
        wk = OMEGA_GPS1 * k
        if wk <= 28:
            ax2.axvline(wk, color='C0', lw=0.5, ls=':', alpha=0.3)
    ax2.set_xlabel(r'Noise peak $\omega_{\rm peak}$ (rad s$^{-1}$)')
    ax2.set_ylabel(r'$\sigma^2_\nu$')
    ax2.set_title(
        rf'$\sigma^2_\nu$ under Lorentzian noise ($\Gamma={width}$)'
        f'\n({opt_label}-noise opt., lower is better)')
    ax2.set_xlim([0, 28])
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(which='both', top=True, right=True)
    ax2.legend()
    fig2.tight_layout()
    for ext in ['pdf', 'png']:
        path = OUTPUT_DIR / f'hf_noise_lorentzian{file_suffix}.{ext}'
        fig2.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')

    plt.close('all')

    # Print tables
    key_cutoffs = [0.0, 1.0, 4.0, 8.0, 16.0]
    print(f'\n--- sigma_nu (high-pass sweep) [{opt_label} opt] ---')
    hdr = f'{"Protocol":<34} ' + ' '.join(f'ω_c={wc:>4.0f}' for wc in key_cutoffs)
    print(hdr); print('-' * len(hdr))
    for label, *_ in protocols:
        d = data[label]
        vals = [sigma_nu_highpass(d['freqs'], d['Fe'], d['sens_sq'], wc) for wc in key_cutoffs]
        print(f'{label:<34} ' + ' '.join(f'{v:>10.4e}' for v in vals))

    print(f'\n--- sigma_nu (Lorentzian at GPS m=8 harmonics) [{opt_label} opt] ---')
    hdr2 = f'{"Protocol":<34} ' + ' '.join(f'ω={wk:>4.0f}' for wk in [8, 16, 24])
    print(hdr2); print('-' * len(hdr2))
    for label, *_ in protocols:
        d = data[label]
        vals = [sigma_nu_lorentzian(d['freqs'], d['Fe'], d['sens_sq'], float(wk), width=width)
                for wk in [8, 16, 24]]
        print(f'{label:<34} ' + ' '.join(f'{v:>10.4e}' for v in vals))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()

    print('Building fixed sequences ...')
    seq_R   = build_ramsey(system)
    seq_G1  = build_gps(system, OMEGA_GPS1)
    seq_G8  = build_gps(system, OMEGA_GPS8)
    seq_dpi = build_double_pi_qsp(system)

    eq_caches = sorted(OUTPUT_DIR.glob('equiangular_opt_cache_*.npz'))
    if not eq_caches:
        raise FileNotFoundError(f'No equiangular_opt_cache_*.npz in {OUTPUT_DIR}')
    eq_cache = np.load(eq_caches[-1], allow_pickle=True)
    print(f'Equiangular cache: {eq_caches[-1].name}')

    qsp_caches = sorted(OUTPUT_DIR.glob('qsp_opt_cache_*.npz'))
    qsp_cache  = np.load(qsp_caches[-1], allow_pickle=True) if qsp_caches else None
    if qsp_cache is not None:
        print(f'QSP cache:         {qsp_caches[-1].name}')
    else:
        print('WARNING: no QSP cache — QSP n>2 omitted.')

    freqs_ana = np.logspace(np.log10(OMEGA_ANA_MIN), np.log10(OMEGA_ANA_MAX), N_ANA)

    for eq_key, qsp_label, file_suffix, opt_label in NOISE_VARIANTS:
        print(f'\n{"="*60}')
        print(f'Noise variant: {opt_label}  (eq_key={eq_key}, suffix="{file_suffix}")')
        print(f'{"="*60}')

        protocols = _build_protocols(system, seq_R, seq_G1, seq_G8, seq_dpi,
                                     eq_cache, qsp_cache, eq_key, qsp_label)

        data = {}
        for label, seq, color, ls, lw in protocols:
            _, sens_sq = detuning_sensitivity(seq)
            _, Fe = analytic_filter(seq, freqs_ana, m_y=1.0)
            data[label] = dict(sens_sq=sens_sq, freqs=freqs_ana, Fe=Fe,
                               color=color, ls=ls, lw=lw)
            snu_w = float(simpson(Fe, x=freqs_ana) / (2 * np.pi)) / sens_sq
            print(f'  {label:<32}  sens={sens_sq:.4f}  sigma_nu_w={snu_w:.4e}')

        _make_figures(protocols, data, file_suffix, opt_label)


if __name__ == '__main__':
    main()
