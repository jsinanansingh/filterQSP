"""
Kubo linear-response validity test (Section VI).

Compares the linear-response filter-function variance prediction against
time-domain Monte Carlo simulation for the ACTUAL sequences used in the code:

  1. Ramsey with finite pi/2 pulses at Omega_fast
  2. GPS m=1 (single continuous x-drive over T)

White noise model:
    H_noise = beta(t) * |e><e|,
    E[beta(t) beta(t')] = S0 * delta(t - t'),
    epsilon_rms = sqrt(S0 * T)

For reference, the script also retains the ideal instantaneous-pulse Ramsey
formula, but the primary validation is now sequence-matched Monte Carlo vs the
filter-function Kubo integral computed from the same finite-duration protocol.
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

def kubo_integral_onesided(seq, omega_max=300.0, n_omega=20000):
    """
    Integrate Fe(omega) against a one-sided white-noise PSD.

    For the convention
        E[beta(t) beta(t')] = S0 delta(t-t'),
    the corresponding one-sided white PSD gives
        Var[M] = S0 * int_0^inf Fe(omega) domega / pi.

    We evaluate the coefficient using the analytic filter rather than the FFT
    grid, because the FFT estimate is sensitive to the coarse treatment of the
    zero-frequency bin for white-noise integrals.
    """
    omegas = np.linspace(0.0, omega_max, n_omega)
    _, Fe = analytic_filter(seq, omegas, m_y=1.0)
    return float(simpson(Fe, x=omegas) / np.pi)


# =============================================================================
# Exact analytic variance for Ramsey (instantaneous-pulse limit)
# =============================================================================

def ramsey_exact_var(S0_arr, T_free):
    """Exact Var[M] = (1 - exp(-2*S0*T_free)) / 8 for M = -sin(Phi)/2."""
    return (1.0 - np.exp(-2.0 * S0_arr * T_free)) / 8.0


# =============================================================================
# Monte Carlo variance via 2x2 time-stepping of the g-e subspace
# =============================================================================

def _control_schedule(seq, dt_target):
    """Return piecewise-constant (omega, delta, dt) arrays on an adaptive grid."""
    omega_steps = []
    delta_steps = []
    dt_steps = []

    for elem in seq.elements:
        dur = elem.duration()
        if dur <= 0.0:
            continue

        n_sub = max(1, int(np.ceil(dur / dt_target)))
        dt = dur / n_sub

        if hasattr(elem, 'omega'):
            omega = float(elem.omega)
            delta = float(elem.delta)
        elif hasattr(elem, 'delta'):
            omega = 0.0
            delta = float(elem.delta)
        else:
            raise ValueError(f'Unsupported element type for MC validation: {type(elem).__name__}')

        omega_steps.extend([omega] * n_sub)
        delta_steps.extend([delta] * n_sub)
        dt_steps.extend([dt] * n_sub)

    return (np.asarray(omega_steps, dtype=float),
            np.asarray(delta_steps, dtype=float),
            np.asarray(dt_steps, dtype=float))

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

    u00 = ph * c
    u01 = np.zeros_like(bk, dtype=complex)
    u11 = ph * c

    mask = OmR > 1e-15
    if np.any(mask):
        ratio_b = bk[mask] / OmR[mask]
        ratio_o = Omega / OmR[mask]
        u00[mask] = ph[mask] * (c[mask] + 1j * ratio_b * s[mask])
        u01[mask] = ph[mask] * (-1j * ratio_o * s[mask])
        u11[mask] = ph[mask] * (c[mask] - 1j * ratio_b * s[mask])
    return u00, u01, u11


def mc_sequence_var(seq, S0, dt_target, n_mc, rng, batch_size=4096):
    """
    Monte Carlo variance of M = sigma_y^{gm} for an actual probe sequence.

    The probe {|g>,|e>} subspace is evolved with piecewise-constant control and
    white noise beta(t)|e><e|.  The clock reference |m> stays fixed at 1/sqrt(2),
    so the final observable is

        M = 2 Im(conj(psi_g) * psi_m),   psi_m = 1/sqrt(2).
    """
    omega_grid, delta_grid, dt_grid = _control_schedule(seq, dt_target)

    count = 0
    sum_M = 0.0
    sum_M2 = 0.0

    while count < n_mc:
        n_batch = min(batch_size, n_mc - count)
        psi_g = np.full(n_batch, 1.0 / np.sqrt(2), dtype=complex)
        psi_e = np.zeros(n_batch, dtype=complex)

        for omega_k, delta_k, dt_k in zip(omega_grid, delta_grid, dt_grid):
            beta_k = delta_k + rng.normal(0.0, np.sqrt(S0 / dt_k), n_batch)
            u00, u01, u11 = _step_u2(beta_k, omega_k, dt_k)
            new_g = u00 * psi_g + u01 * psi_e
            psi_e = u01 * psi_g + u11 * psi_e
            psi_g = new_g

        M = 2.0 * np.imag(np.conj(psi_g) / np.sqrt(2))
        sum_M += float(np.sum(M))
        sum_M2 += float(np.sum(M**2))
        count += n_batch

    mean_M = sum_M / n_mc
    return sum_M2 / n_mc - mean_M**2


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
    print('Kubo integrals (one-sided, actual sequences):')
    print(f'  Ramsey finite-pulse  k = {k_R:.4f}   (instantaneous-limit T_free/4 = {tau_free/4:.4f})')
    print(f'  GPS m=1             k = {k_G1:.4f}')

    # Sweep epsilon_rms = sqrt(S0 * T)
    eps_arr = np.logspace(np.log10(0.05), np.log10(3.5), 50)
    S0_arr  = eps_arr**2 / T

    # --- Ramsey ---
    var_R_exact_inst = ramsey_exact_var(S0_arr, tau_free)
    var_R_kubo_inst  = S0_arr * tau_free / 4.0      # instantaneous-pulse limit
    var_R_kubo_seq   = S0_arr * k_R                 # actual finite-pulse sequence

    # MC only at a coarser grid to save time
    eps_mc = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0, 2.8, 3.5])
    S0_mc  = eps_mc**2 / T
    var_R_mc_inst = np.array([mc_ramsey_var(s, tau_free, 200_000, rng) for s in S0_mc])
    dt_target_R = tau_free / 1500.0
    n_mc_R    = 30_000
    print(f'Running Ramsey finite-pulse MC ({len(eps_mc)} noise levels, {n_mc_R} samples each)...')
    var_R_mc_seq = np.array([
        mc_sequence_var(seq_R, s, dt_target_R, n_mc_R, rng) for s in S0_mc
    ])
    print('  done.')

    # --- GPS m=1 ---
    dt_target_G1 = T / 1200.0
    n_mc    = 30_000
    print(f'\nRunning GPS m=1 MC ({len(eps_mc)} noise levels, {n_mc} samples each)...')
    var_G1_mc = np.array([
        mc_sequence_var(seq_G1, s, dt_target_G1, n_mc, rng) for s in S0_mc
    ])
    print('  done.')

    # Kubo for GPS m=1 (actual sequence and ideal analytic reference)
    var_G1_kubo_seq = S0_arr * k_G1
    slope_G1_inst   = 3.0 * np.pi / 4.0
    var_G1_kubo_inst = S0_arr * slope_G1_inst

    # ==========================================================================
    # Figure 1: absolute variance vs epsilon_rms
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    ax.loglog(eps_arr, var_R_exact_inst, '0.5',  lw=1.5, label='Ramsey exact (instantaneous limit)')
    ax.loglog(eps_arr, var_R_kubo_inst,  '0.5',  lw=1.2, ls='--', label=r'Ramsey Kubo (instantaneous limit)')
    ax.loglog(eps_arr, var_R_kubo_seq,   'C3--', lw=2.0, label='Ramsey Kubo (actual sequence)')
    ax.loglog(eps_mc,  var_R_mc_seq,     'C3o',  ms=5,   label='Ramsey MC (actual sequence)')
    ax.loglog(eps_mc,  var_R_mc_inst,    'o', color='0.5', ms=4, label='Ramsey MC (instantaneous limit)')
    ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\epsilon_{\rm rms} = \sqrt{S_0 T}$', fontsize=11)
    ax.set_ylabel(r'${\rm Var}[M]$', fontsize=11)
    ax.set_title('Ramsey', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.04, 4])
    ax.set_ylim([5e-5, 0.5])

    ax = axes[1]
    ax.loglog(eps_arr, var_G1_kubo_inst, '0.5',  lw=1.2, label=r'GPS m=1 Kubo (ideal $\frac{3\pi}{4}S_0$)')
    ax.loglog(eps_arr, var_G1_kubo_seq,  'C0--', lw=2.0, label='GPS m=1 Kubo (actual sequence)')
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
    ratio_R_exact_inst = var_R_exact_inst / (S0_arr * tau_free / 4.0)
    ratio_R_mc_inst    = var_R_mc_inst    / (S0_mc   * tau_free / 4.0)
    ratio_R_mc_seq     = var_R_mc_seq     / (S0_mc   * k_R)

    ratio_G1_mc = var_G1_mc / (S0_mc * k_G1)

    fig2, ax2 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2.semilogx(eps_ratio, ratio_R_exact_inst,  '0.5', lw=1.8, label='Ramsey exact / Kubo$_0$ (instantaneous)')
    ax2.semilogx(eps_mc,    ratio_R_mc_inst,     'o', color='0.5', ms=5, label='Ramsey MC / Kubo$_0$ (instantaneous)')
    ax2.semilogx(eps_mc,    ratio_R_mc_seq,      'C3o',  ms=6,   label='Ramsey MC / Kubo$_0$ (actual sequence)')
    ax2.semilogx(eps_mc,    ratio_G1_mc,         'C0s',  ms=6,   label='GPS m=1 MC / Kubo$_0$ (actual sequence)')
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
    # Show sequence-matched Kubo slope vs MC for BOTH protocols on one panel.
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes3[0]
    ax.loglog(eps_arr, var_R_kubo_seq, 'C3--', lw=2, label='Kubo (filter function, actual sequence)')
    ax.loglog(eps_mc,  var_R_mc_seq,   'C3o',  ms=5, label='MC (actual sequence)')
    ax.loglog(eps_arr, var_R_kubo_inst, '0.5', lw=1.2, label='Kubo (instantaneous limit)')
    ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\epsilon_{\rm rms}$', fontsize=11)
    ax.set_ylabel(r'${\rm Var}[M]$', fontsize=11)
    ax.set_title('Ramsey: Kubo vs MC', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.04, 4])
    ax.set_ylim([5e-5, 0.5])

    ax = axes3[1]
    ax.loglog(eps_arr, var_G1_kubo_seq, 'C0--', lw=2, label='Kubo (filter function, actual sequence)')
    ax.loglog(eps_arr, var_G1_kubo_inst, '0.5', lw=1.2, label=r'Kubo (ideal $\frac{3\pi}{4}S_0$)')
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
        r'Filter-function Kubo formula vs time-domain Monte Carlo ($T=2\pi$, white noise)',
        fontsize=12)

    for ext in ['pdf', 'png']:
        fig3.savefig(OUTPUT_DIR / f'kubo_3level.{ext}', dpi=300, bbox_inches='tight')
    print('Saved kubo_3level')

    # ==========================================================================
    # Summary table
    # ==========================================================================
    print('\n--- Kubo breakdown summary ---')
    print(f'{"eps_rms":>10}  {"Var_R_MC_seq":>14}  {"Var_R_Kubo_seq":>14}  {"ratio":>8}')
    for eps, S0, v_ex, v_ku in zip(
            eps_mc, S0_mc, var_R_mc_seq, S0_mc * k_R):
        print(f'{eps:>10.3f}  {v_ex:>14.5e}  {v_ku:>12.5e}  {v_ex/v_ku:>8.4f}')

    plt.close('all')


if __name__ == '__main__':
    main()
