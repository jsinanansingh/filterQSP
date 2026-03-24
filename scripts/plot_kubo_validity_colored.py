"""
Kubo validity test for 1/f and 1/f^2 colored noise.

Confirms that the Kubo linear-response threshold epsilon_rms ~ 0.2 holds
for coloured noise by comparing two independent quantities:

  1. Analytic exact Ramsey formula (all-orders, derived from Gaussian Phi):
       Var[M]_exact = (1 - exp(-8*kubo_R)) / 8
     where sigma_phi^2 = 4*kubo_R is inferred from the filter-function Kubo
     integral kubo_R = int Fe(w)*S(w) dw/pi.

  2. Monte Carlo (direct numerical experiment, colour-independent):
       - Generate coloured noise beta(t) via spectral method
       - Accumulate phase Phi = int_0^T beta(t) dt (Ramsey free evolution)
       - M = -sin(Phi)/2
       - Empirical Var[M] from N_MC trajectories

The MC ratio Var[M]_MC / kubo_R independently validates the Kubo formula
AND the analytic exact formula for each noise colour.

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
N_STEPS    = 1000          # time steps for MC propagation
N_MC       = 8_000         # MC trajectories per noise level
BATCH_SIZE = 200           # trajectories per batch
K_OVER     = 50            # oversampling factor for spectral noise generation
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'

dt        = T / N_STEPS
omega_min = 2 * np.pi / T   # Fourier-limit lower cutoff (= 1.0 for T=2pi)
omega_max = np.pi / dt       # Nyquist upper limit


# =============================================================================
# Sequence builder (for filter-function Kubo integral only)
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
# Kubo prediction: one-sided integral with /pi normalization
# =============================================================================

def kubo_colored(seq, S0, alpha, n_omega=30_000):
    """
    Kubo variance = int_{omega_min}^{omega_max} Fe(w) * S0/w^alpha dw / pi.

    The /pi factor (not /(2pi)) correctly accounts for the convention:
    Var[M] = 2 * int_0^inf Fe(w) * S_2(w) dw/(2pi)
    where the two-sided PSD S_2(w) = S0/(2*w^alpha) satisfies S_1 = 2*S_2,
    giving int_0^inf Fe * S_1(w) dw / (2pi) = int_0^inf Fe * S0/w^alpha dw / pi.

    Consistent with kubo_integral_onesided in plot_kubo_validity.py (white noise),
    and verified against MC for white noise (ratio = 1 in linear regime).
    """
    omegas = np.linspace(omega_min * 1.001, omega_max, n_omega)
    _, Fe = analytic_filter(seq, omegas, m_y=1.0)
    S = S0 / omegas ** alpha
    return float(simpson(Fe * S, x=omegas) / np.pi)


# =============================================================================
# Analytic exact Ramsey formula (all-orders)
# =============================================================================

def ramsey_exact_var(kubo_R):
    """
    For M = -sin(Phi)/2, Phi Gaussian with sigma_phi^2 = 4*kubo_R:
        Var[M]_exact = (1 - exp(-2*sigma_phi^2)) / 8
                     = (1 - exp(-8*kubo_R)) / 8.
    Valid for any noise colour since Phi is a linear Gaussian integral.
    In the linear regime: approx kubo_R  =>  ratio = 1. ✓
    """
    return (1.0 - np.exp(-8.0 * kubo_R)) / 8.0


# =============================================================================
# Coloured noise generation via spectral method (oversampled grid)
# =============================================================================

def make_colored_noise_batch(S0, alpha, n_batch, rng):
    """
    Generate n_batch realisations of bandlimited coloured noise with
    one-sided PSD S_1(omega) = S0/omega^alpha for omega in [omega_min, omega_max].

    Uses K_OVER * N_STEPS oversampling to resolve the noise spectrum finely
    near omega_min.  The first N_STEPS points of the longer time series are
    returned: for a stationary process, any T-length window has the same
    single-shot statistics.

    The spectral method assigns psd(omega_j) = S0/omega_j^alpha to positive
    frequency bins of the rfft.  For a real process this corresponds to
    two-sided PSD S_2 = psd/2, so the resulting Var[M] satisfies:
        Var[M] = 2 * int_0^inf Fe * S_2 dw/(2pi) = int_0^inf Fe * psd dw / pi
    which matches kubo_colored (with /pi normalisation).

    Returns array of shape (n_batch, N_STEPS).
    """
    N_long = K_OVER * N_STEPS

    white = rng.normal(0.0, 1.0 / np.sqrt(dt), (n_batch, N_long))
    white_fft = np.fft.rfft(white, axis=1)

    freqs_long = np.fft.rfftfreq(N_long, dt)
    omega_long = 2 * np.pi * freqs_long

    with np.errstate(divide='ignore', invalid='ignore'):
        psd = np.where(omega_long > omega_min, S0 / (omega_long ** alpha), 0.0)
    psd[0] = 0.0

    scaled_fft = white_fft * np.sqrt(psd)[np.newaxis, :]
    beta_long = np.fft.irfft(scaled_fft, n=N_long, axis=1)
    return beta_long[:, :N_STEPS]


# =============================================================================
# Monte Carlo: Ramsey with coloured noise (independent of analytic formula)
# =============================================================================

def mc_ramsey_colored(S0, alpha, rng, n_mc=N_MC, batch_size=BATCH_SIZE):
    """
    Direct Monte Carlo of Var[M] for Ramsey under coloured noise.

    In the instantaneous-pulse limit (Omega_fast -> inf), the signal is
    M = -sin(Phi)/2 where Phi = int_0^T beta(t) dt is the accumulated phase.
    We compute Phi = sum(beta_k * dt) from the coloured noise realisations and
    return the empirical variance of M.  This is independent of the analytic
    exact formula and of the filter-function Kubo integral.

    The comparison Var[M]_MC / kubo_R = 1 in the linear regime simultaneously
    validates (i) the spectral noise generation normalisation, (ii) the Kubo
    integral formula, and (iii) the exact Ramsey variance formula.
    """
    sum_M  = 0.0
    sum_M2 = 0.0
    count  = 0

    while count < n_mc:
        n_batch = min(batch_size, n_mc - count)
        beta = make_colored_noise_batch(S0, alpha, n_batch, rng)

        # Phi = integral of beta over the free evolution ≈ T
        # (correction from finite pi/2 pulses is O(tau_pi2/T) ~ 0.001, negligible)
        Phi = np.sum(beta, axis=1) * dt   # shape (n_batch,)

        M = -0.5 * np.sin(Phi)
        sum_M  += float(np.sum(M))
        sum_M2 += float(np.sum(M ** 2))
        count  += n_batch

    mean_M = sum_M / n_mc
    return sum_M2 / n_mc - mean_M ** 2


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7777)
    system = ThreeLevelClock()
    seq_R = build_ramsey(system)

    S0_arr = np.logspace(-2, 2, 18)

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
        eps_arr    = 2.0 * np.sqrt(kubo_R_arr)

        # Analytic exact Ramsey variance
        var_R_exact = ramsey_exact_var(kubo_R_arr)

        # MC Ramsey variance (independent numerical experiment)
        print(f'  Running Ramsey MC ({len(S0_arr)} noise levels, '
              f'{N_MC} trajectories each)...')
        var_R_mc = []
        for i, s in enumerate(S0_arr):
            v = mc_ramsey_colored(s, alpha, rng)
            var_R_mc.append(v)
            ratio_exact = var_R_exact[i] / kubo_R_arr[i]
            ratio_mc    = v / kubo_R_arr[i]
            print(f'    S0={s:.3e}  eps={eps_arr[i]:.3f}'
                  f'  ratio_exact={ratio_exact:.4f}  ratio_MC={ratio_mc:.4f}')
        var_R_mc = np.array(var_R_mc)

        all_results[alpha] = dict(
            eps        = eps_arr,
            kubo_R     = kubo_R_arr,
            var_exact  = var_R_exact,
            var_mc     = var_R_mc,
            ratio_exact= var_R_exact / kubo_R_arr,
            ratio_mc   = var_R_mc   / kubo_R_arr,
        )

    # =========================================================================
    # Figure
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    for ax, (alpha, label) in zip(axes, noise_models):
        r = all_results[alpha]

        ax.semilogx(r['eps'], r['ratio_exact'], 'C3-',  lw=1.8,
                    label='Ramsey (exact)')
        ax.semilogx(r['eps'], r['ratio_mc'],    'C3o',  ms=4,
                    label='Ramsey MC', lw=0.0)

        ax.axhline(1.0,  color='gray', lw=0.9, ls='--')
        ax.axvline(0.2,  color='k',    lw=0.8, ls=':',  alpha=0.6,
                   label=r'$\epsilon_{\rm rms}=0.2$')
        ax.axhline(0.95, color='gray', lw=0.6, ls=':', alpha=0.4)

        ax.set_xlabel(r'$\epsilon_{\rm rms}$')
        ax.set_ylabel(
            r'$\mathrm{Var}_{\rm exact}\;/\;\mathrm{Var}_{\rm Kubo}$')
        ax.set_title(label + ' noise')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.25, which='both')
        ax.tick_params(which='both', top=True, right=True)
        ax.set_xlim([0.04, 6])
        ax.set_ylim([0, 1.4])

    fig.suptitle(
        r'Kubo validity: exact and MC variance vs Kubo, $T=2\pi$')

    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'kubo_colored_noise.{ext}',
                    dpi=300, bbox_inches='tight')
    print(f'\nSaved kubo_colored_noise.{{pdf,png}}')

    # =========================================================================
    # Summary table
    # =========================================================================
    print('\n--- Summary: ratios at eps_rms ~ 0.2 ---')
    for alpha, label in noise_models:
        r = all_results[alpha]
        idx = np.argmin(np.abs(r['eps'] - 0.2))
        print(f'  {label}  eps={r["eps"][idx]:.3f}'
              f'  ratio_exact={r["ratio_exact"][idx]:.4f}'
              f'  ratio_MC={r["ratio_mc"][idx]:.4f}')

    plt.close('all')


if __name__ == '__main__':
    main()
