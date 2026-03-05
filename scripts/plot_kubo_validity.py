"""
Kubo linear-response validity test (Section VI).

Compares the linear-response (Kubo) variance prediction against the exact
result for two protocols as a function of noise amplitude epsilon_rms.

White noise model:  H_noise = beta(t) * |e><e|,
                    E[beta(t) beta(t')] = S0 * delta(t - t')
                    epsilon_rms = sqrt(S0 * T)

Analytic linear-Kubo slopes (derived by first-order perturbation theory):

  Ramsey (instantaneous-pulse limit):
    M = -sin(Phi)/2,   Phi = integral_0^T beta dt ~ N(0, S0*T)
    Var_exact[M] = (1 - exp(-2*S0*T)) / 8           (analytic, all orders)
    Var_Kubo[M]  = S0 * T / 4                        (linear in S0)

  GPS m=1 (Omega=1, T=2pi, single continuous x-drive):
    delta_M = -integral_0^T beta(t) sin(Omega*(T-t)/2) sin(Omega*t/2) dt
    For Omega*T=2pi: sin(Omega*(T-t)/2)*sin(Omega*t/2) = sin^2(Omega*t/2)
    Var_Kubo[M]  = S0 * integral_0^T sin^4(t/2) dt = S0 * 3*pi/4
    Exact variance via Monte Carlo (full 2x2 g-e propagation).
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
from quantum_pulse_suite.core.three_level_filter import fft_three_level_filter

T          = 2 * np.pi
OMEGA_FAST = 20.0 * np.pi
OMEGA_GPS1 = 1.0          # 1 complete Rabi cycle in T = 2pi
OUTPUT_DIR = Path(__file__).parent.parent / 'figures' / 'qubit_performance_plots'


# =============================================================================
# Sequence builders (same conventions as plot_protocol_comparison.py)
# =============================================================================

def build_ramsey(system):
    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T - 2 * tau_pi2
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.add_free_evolution(tau_free, 0.0)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], 0.0, tau_pi2)
    seq.compute_polynomials()
    return seq, tau_free


def build_gps1(system):
    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_GPS1, [1, 0, 0], 0.0, T)
    seq.compute_polynomials()
    return seq


# =============================================================================
# Kubo prediction from filter function (code convention: one-sided integral)
# =============================================================================

def kubo_integral_onesided(seq):
    """Integrate Fe(omega) over positive frequencies (code's Kubo convention)."""
    fr, Fe, _, _ = fft_three_level_filter(seq, n_samples=4096, pad_factor=4, m_y=1.0)
    return float(simpson(Fe, x=fr) / (2 * np.pi))


# =============================================================================
# Exact analytic variance for Ramsey (instantaneous-pulse limit)
# =============================================================================

def ramsey_exact_var(S0_arr, T_free):
    """Exact Var[M] = (1 - exp(-2*S0*T_free)) / 8 for M = -sin(Phi)/2."""
    return (1.0 - np.exp(-2.0 * S0_arr * T_free)) / 8.0


# =============================================================================
# Monte Carlo variance via 2x2 time-stepping of the g-e subspace
# =============================================================================

def _step_u2(bk, Omega, dt):
    """
    Propagator U = exp(-i * H * dt) for H = [[0, Omega/2], [Omega/2, beta]] (2x2).

    Returns (u00, u01, u11) arrays of shape (n_mc,).
    U is symmetric: u10 = u01.
    """
    OmR = np.sqrt(Omega**2 + bk**2)
    half = OmR * dt / 2.0
    c    = np.cos(half)
    s    = np.sin(half)
    ph   = np.exp(-1j * bk * dt / 2.0)
    u00  = ph * (c + 1j * bk / OmR * s)
    u01  = ph * (-1j * Omega / OmR * s)
    u11  = ph * (c - 1j * bk / OmR * s)
    return u00, u01, u11


def mc_gps_var(S0, Omega, n_steps, n_mc, rng):
    """
    Monte Carlo variance of M = 2 Im(conj(psi_g) * 1/sqrt(2)) for GPS.

    Initial g-e state: (1/sqrt(2), 0).  |m> stays at 1/sqrt(2).
    """
    dt        = T / n_steps
    sig_beta  = np.sqrt(S0 / dt)
    beta_all  = rng.normal(0.0, sig_beta, (n_mc, n_steps))

    psi_g = np.full(n_mc, 1.0 / np.sqrt(2), dtype=complex)
    psi_e = np.zeros(n_mc, dtype=complex)

    for k in range(n_steps):
        u00, u01, u11 = _step_u2(beta_all[:, k], Omega, dt)
        new_g  = u00 * psi_g + u01 * psi_e
        psi_e  = u01 * psi_g + u11 * psi_e
        psi_g  = new_g

    # M = 2 * Im(conj(psi_g) * psi_m),  psi_m = 1/sqrt(2)
    M = 2.0 * np.imag(np.conj(psi_g) / np.sqrt(2))
    return float(np.var(M))


def mc_ramsey_var(S0, T_free, n_mc, rng):
    """
    Monte Carlo variance for Ramsey using the exact formula (instant-pulse limit).

    Draws Phi ~ N(0, S0*T_free) and computes Var[-sin(Phi)/2].
    """
    Phi = rng.normal(0.0, np.sqrt(S0 * T_free), n_mc)
    return float(np.var(-np.sin(Phi) / 2.0))


# =============================================================================
# Main sweep
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng    = np.random.default_rng(12345)
    system = ThreeLevelClock()

    seq_R, tau_free = build_ramsey(system)
    seq_G1          = build_gps1(system)

    k_R  = kubo_integral_onesided(seq_R)
    k_G1 = kubo_integral_onesided(seq_G1)
    print(f'Kubo integrals (one-sided, code convention):')
    print(f'  Ramsey  k = {k_R:.4f}   (T/4 = {T/4:.4f})')
    print(f'  GPS m=1 k = {k_G1:.4f}')

    # Sweep epsilon_rms = sqrt(S0 * T)
    eps_arr = np.logspace(np.log10(0.05), np.log10(3.5), 50)
    S0_arr  = eps_arr**2 / T

    # --- Ramsey ---
    var_R_exact  = ramsey_exact_var(S0_arr, tau_free)
    var_R_kubo   = S0_arr * tau_free / 4.0          # correct linear Kubo (exact slope)
    var_R_kubo_c = S0_arr * k_R                     # code's one-sided Kubo (may differ)

    # MC only at a coarser grid to save time
    eps_mc = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0, 2.8, 3.5])
    S0_mc  = eps_mc**2 / T
    var_R_mc  = np.array([mc_ramsey_var(s, tau_free, 200_000, rng) for s in S0_mc])

    # --- GPS m=1 ---
    n_steps = 800
    n_mc    = 30_000
    print(f'\nRunning GPS m=1 MC ({len(eps_mc)} noise levels, {n_mc} samples each)...')
    var_G1_mc = np.array([
        mc_gps_var(s, OMEGA_GPS1, n_steps, n_mc, rng) for s in S0_mc
    ])
    print('  done.')

    # Code Kubo for GPS (one-sided convention)
    var_G1_kubo_c = S0_arr * k_G1

    # Analytic linear-Kubo slope for GPS m=1: int_0^{2pi} sin^4(t/2) dt = 3*pi/4
    slope_G1      = 3.0 * np.pi / 4.0
    var_G1_kubo_a = S0_arr * slope_G1          # two-sided, correct analytic Kubo

    # ==========================================================================
    # Figure 1: absolute variance vs epsilon_rms
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    ax.loglog(eps_arr, var_R_exact,  'C3',   lw=2,   label='Ramsey exact')
    ax.loglog(eps_arr, var_R_kubo,   'C3--', lw=1.5, label=r'Ramsey Kubo ($S_0 T/4$)')
    ax.loglog(eps_arr, var_R_kubo_c, 'C3:',  lw=1.5, label=r'Ramsey Kubo (code $k$)')
    ax.loglog(eps_mc,  var_R_mc,     'C3o',  ms=5,   label='Ramsey MC')
    ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\epsilon_{\rm rms} = \sqrt{S_0 T}$', fontsize=11)
    ax.set_ylabel(r'${\rm Var}[M]$', fontsize=11)
    ax.set_title('Ramsey', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.04, 4])
    ax.set_ylim([5e-5, 0.5])

    ax = axes[1]
    ax.loglog(eps_arr, var_G1_kubo_a, 'C0-',  lw=2.0, label=r'GPS m=1 Kubo ($\frac{3\pi}{4}S_0$)')
    ax.loglog(eps_arr, var_G1_kubo_c, 'C0--', lw=1.5, label='GPS m=1 Kubo (code, 1-sided)')
    ax.loglog(eps_mc,  var_G1_mc,     'C0o',  ms=5,   label='GPS m=1 MC')
    ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\epsilon_{\rm rms} = \sqrt{S_0 T}$', fontsize=11)
    ax.set_ylabel(r'${\rm Var}[M]$', fontsize=11)
    ax.set_title('GPS m=1', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.04, 4])
    ax.set_ylim([5e-5, 0.5])

    fig.suptitle(
        r'Exact vs Kubo variance: white noise $H_\beta = \beta(t)|e\rangle\langle e|$, $T=2\pi$',
        fontsize=12)

    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'kubo_2level.{ext}', dpi=300, bbox_inches='tight')
    print('Saved kubo_2level')

    # ==========================================================================
    # Figure 2: ratio Var_exact / Var_Kubo_linear vs epsilon_rms
    # ==========================================================================
    # Use the *correct* Kubo slope (T_free/4 for Ramsey) as the linear baseline.
    # The ratio is 1 in the linear regime and decreases when Kubo overestimates.
    eps_ratio = eps_arr
    ratio_R_exact = var_R_exact / (S0_arr * tau_free / 4.0)
    ratio_R_mc    = var_R_mc     / (S0_mc   * tau_free / 4.0)

    ratio_G1_mc = var_G1_mc / (S0_mc * slope_G1)

    fig2, ax2 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2.semilogx(eps_ratio, ratio_R_exact,  'C3',   lw=2.5, label='Ramsey exact / Kubo$_0$')
    ax2.semilogx(eps_mc,    ratio_R_mc,     'C3o',  ms=6,   label='Ramsey MC / Kubo$_0$')
    ax2.semilogx(eps_mc,    ratio_G1_mc,    'C0s',  ms=6,   label='GPS m=1 MC / Kubo$_0$')
    ax2.axhline(1.0, color='gray', lw=1.2, ls='--', label='Kubo (linear response)')
    ax2.axvline(1.0, color='k',    lw=0.8, ls='--', alpha=0.5, label=r'$\epsilon_{\rm rms}=1$')
    ax2.set_xlabel(r'$\epsilon_{\rm rms} = \sqrt{S_0 T}$', fontsize=12)
    ax2.set_ylabel(r'${\rm Var}_{\rm exact}\;/\;{\rm Var}_{\rm Kubo}^{(0)}$', fontsize=12)
    ax2.set_title(
        r'Kubo validity: ratio of exact to linear-response variance'
        '\n'
        r'Ratio $\to 1$ for $\epsilon_{\rm rms}\ll 1$; deviation marks nonlinear regime',
        fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([0.04, 4])
    ax2.set_ylim([0, 1.4])

    for ext in ['pdf', 'png']:
        fig2.savefig(OUTPUT_DIR / f'kubo_filter_comparison.{ext}', dpi=300, bbox_inches='tight')
    print('Saved kubo_filter_comparison')

    # ==========================================================================
    # Figure 3: filter-function Kubo vs MC comparison
    # ==========================================================================
    # Show code-Kubo slope vs MC for BOTH protocols on one panel.
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes3[0]
    ax.loglog(eps_arr, var_R_kubo_c, 'C3--', lw=2, label='Kubo (filter function)')
    ax.loglog(eps_arr, var_R_exact,  'C3',   lw=2, label='Exact analytic')
    ax.loglog(eps_mc,  var_R_mc,     'C3o',  ms=5, label='MC')
    ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\epsilon_{\rm rms}$', fontsize=11)
    ax.set_ylabel(r'${\rm Var}[M]$', fontsize=11)
    ax.set_title('Ramsey: Kubo vs exact', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.04, 4])
    ax.set_ylim([5e-5, 0.5])

    ax = axes3[1]
    ax.loglog(eps_arr, var_G1_kubo_a, 'C0-',  lw=2, label=r'Kubo analytic ($\frac{3\pi}{4}S_0$)')
    ax.loglog(eps_arr, var_G1_kubo_c, 'C0--', lw=2, label='Kubo code (1-sided)')
    ax.loglog(eps_mc,  var_G1_mc,     'C0o',  ms=5, label='MC')
    ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\epsilon_{\rm rms}$', fontsize=11)
    ax.set_ylabel(r'${\rm Var}[M]$', fontsize=11)
    ax.set_title('GPS m=1: Kubo vs MC', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.04, 4])
    ax.set_ylim([5e-5, 0.5])

    fig3.suptitle(
        r'Filter-function Kubo formula vs exact variance ($T=2\pi$, white noise)',
        fontsize=12)

    for ext in ['pdf', 'png']:
        fig3.savefig(OUTPUT_DIR / f'kubo_3level.{ext}', dpi=300, bbox_inches='tight')
    print('Saved kubo_3level')

    # ==========================================================================
    # Summary table
    # ==========================================================================
    print('\n--- Kubo breakdown summary ---')
    print(f'{"eps_rms":>10}  {"Var_R_exact":>14}  {"Var_R_Kubo":>12}  {"ratio":>8}')
    for eps, S0, v_ex, v_ku in zip(
            eps_arr[::5], S0_arr[::5], var_R_exact[::5], var_R_kubo[::5]):
        print(f'{eps:>10.3f}  {v_ex:>14.5e}  {v_ku:>12.5e}  {v_ex/v_ku:>8.4f}')

    plt.close('all')


if __name__ == '__main__':
    main()
