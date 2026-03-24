"""
Kubo validity test for 1/f and 1/f^2 colored noise (Ramsey, analytic).

Confirms that the Kubo linear-response approximation breaks down at the same
epsilon_rms threshold (~0.2) regardless of noise colour.

Protocol: Ramsey (instantaneous-pulse limit).  The exact all-orders result
    M = -sin(Phi)/2,  Phi = int_0^{tau_free} beta(t) dt,
is Gaussian regardless of noise colour (linear stochastic integral), so:
    Var[M]_exact = (1 - exp(-2 sigma_phi^2)) / 8,
    sigma_phi^2  = 4 * Var[M]_Kubo   (verified by DC-consistency of Fe).

The Kubo variance for coloured noise:
    Var[M]_Kubo = int_{omega_min}^{omega_max} Fe(omega) S(omega) domega / pi,
where S(omega) = S0 / omega^alpha (one-sided PSD) and Fe is the analytic
3-level filter function from analytic_filter.

Usage:
    python scripts/plot_kubo_validity_colored.py
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
from quantum_pulse_suite.core.three_level_filter import analytic_filter

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.size':          8,
    'axes.labelsize':     9,
    'axes.titlesize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '0.7',
    'lines.linewidth':    1.4,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
})

T          = 2 * np.pi
OMEGA_FAST = 20.0 * np.pi
N_STEPS    = 1000
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'

dt        = T / N_STEPS
omega_min = 2 * np.pi / T   # Fourier-limit lower cutoff (= 1.0 for T=2pi)
omega_max = np.pi / dt       # Nyquist upper limit


# =============================================================================
# Sequence builder
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


# =============================================================================
# Kubo prediction (filter-function integral against coloured PSD)
# =============================================================================

def kubo_colored(seq, S0, alpha, n_omega=30_000):
    """
    Kubo variance: int_0^inf Fe(omega) * S0/omega^alpha domega / pi,
    integrated from omega_min to omega_max.

    The /pi factor (not /(2pi)) correctly accounts for the two-sided PSD
    convention: Var[M] = 2 * int_0^inf Fe * S_2 domega/(2pi) where the
    one-sided PSD S_1(omega) = 2*S_2(omega), so S_2 = S0/(2*omega^alpha)
    and Var[M] = int_0^inf Fe * S0/omega^alpha domega / pi.

    DC consistency: Fe(0) = sens_sq, so for DC noise S=S0:
        Var[M]_Kubo|_{DC} = S0 * sens_sq / pi * (omega_max - omega_min)
    which is the correct single-sided white-noise prediction (matches
    kubo_integral_onesided in plot_kubo_validity.py using /pi).
    """
    omegas = np.linspace(omega_min * 1.001, omega_max, n_omega)
    _, Fe = analytic_filter(seq, omegas, m_y=1.0)
    S = S0 / omegas ** alpha
    return float(simpson(Fe * S, x=omegas) / np.pi)


# =============================================================================
# Ramsey: exact analytic formula
# =============================================================================

def ramsey_exact_var(kubo_R):
    """
    For the 3-level Ramsey sequence, M = -sin(Phi)/2 exactly, where
    Phi = integral_0^{tau_free} beta(t) dt is Gaussian with
    sigma_phi^2 = 4 * kubo_R  (from DC-consistency: Fe^Ramsey = sin^2(w*T/2)/w^2
    and Var[Phi] = int_{-inf}^inf S_2(w) * 4sin^2(wT/2)/w^2 dw/(2pi) = 4*kubo_R).

    The exact all-orders variance is:
        Var[M]_exact = (1 - exp(-2 sigma_phi^2)) / 8
                     = (1 - exp(-8 * kubo_R)) / 8

    In the linear regime (kubo_R << 1):
        Var[M]_exact -> kubo_R  (since (1-(1-8k))/8 = k)  ✓

    Valid for any noise color since Phi is always Gaussian.
    """
    sigma_phi_sq = 4.0 * kubo_R
    return (1.0 - np.exp(-2.0 * sigma_phi_sq)) / 8.0


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system = ThreeLevelClock()
    seq_R = build_ramsey(system)

    # S0 values: log-spaced; compute eps_rms numerically for x-axis
    S0_arr = np.logspace(-2, 2, 22)

    noise_models = [
        (1, r'$1/f$'),
        (2, r'$1/f^2$'),
    ]

    all_results = {}

    for alpha, label in noise_models:
        print(f'\n{"=" * 60}')
        print(f'Noise model: {label}  (alpha={alpha})')
        print(f'{"=" * 60}')

        print('  Computing Kubo integrals...')
        kubo_R_arr = np.array([kubo_colored(seq_R, s, alpha) for s in S0_arr])

        # eps_rms = sqrt(sigma_phi^2) = 2*sqrt(kubo_R)
        eps_arr = 2.0 * np.sqrt(kubo_R_arr)

        # Exact Ramsey variance (analytic, noise-colour-independent formula)
        var_R_exact = ramsey_exact_var(kubo_R_arr)
        ratio_R = var_R_exact / kubo_R_arr

        for i in range(len(S0_arr)):
            print(f'    S0={S0_arr[i]:.3e}  eps_rms={eps_arr[i]:.3f}'
                  f'  ratio_R={ratio_R[i]:.4f}')

        all_results[alpha] = dict(
            eps=eps_arr,
            var_R_exact=var_R_exact,
            kubo_R=kubo_R_arr,
            ratio_R=ratio_R,
        )

    # =========================================================================
    # Figure: ratio Var_exact / Var_Kubo vs eps_rms, one panel per noise model
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    for ax, (alpha, label) in zip(axes, noise_models):
        r = all_results[alpha]

        ax.semilogx(r['eps'], r['ratio_R'], 'C3-', lw=1.8,
                    label='Ramsey (exact)')

        ax.axhline(1.0,  color='gray', lw=0.9, ls='--')
        ax.axvline(0.2,  color='k',    lw=0.8, ls=':',  alpha=0.6,
                   label=r'$\epsilon_{\rm rms}=0.2$')
        ax.axhline(0.95, color='gray', lw=0.6, ls=':', alpha=0.4)

        ax.set_xlabel(r'$\epsilon_{\rm rms}$')
        ax.set_ylabel(
            r'$\mathrm{Var}_{\mathrm{exact}}\;/\;\mathrm{Var}_{\mathrm{Kubo}}$')
        ax.set_title(label + ' noise')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.25, which='both')
        ax.tick_params(which='both', top=True, right=True)
        ax.set_xlim([0.04, 6])
        ax.set_ylim([0, 1.4])

    fig.suptitle(
        r'Kubo validity: exact vs linear-response variance, $T=2\pi$')

    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'kubo_colored_noise.{ext}',
                    dpi=300, bbox_inches='tight')
    print(f'\nSaved kubo_colored_noise.{{pdf,png}}')

    # =========================================================================
    # Summary table
    # =========================================================================
    print('\n--- Summary: ratio at eps_rms ~ 0.2 ---')
    for alpha, label in noise_models:
        r = all_results[alpha]
        idx = np.argmin(np.abs(r['eps'] - 0.2))
        print(f'  {label}  eps={r["eps"][idx]:.3f}'
              f'  ratio_R={r["ratio_R"][idx]:.4f}')

    plt.close('all')


if __name__ == '__main__':
    main()
