
"""
Compare sigma_y frequency sensitivity: Ramsey, Rabi, and GPS protocols.

Three computation paths are shown for each protocol, distinguished by linestyle:

  Solid   — Analytic (3-level):  polynomial/integral formula for Fe(w)
  Dashed  — FFT (3-level):       fft_three_level_filter – propagates the full
              3-level unitary U(t), projects to the probe subspace, samples
              phi(t)=F*(t)G(t) and FFTs.  Independently validates the analytic.
  Dotted  — FFT (2-level):       fft_filter_function on a qubit – propagates
              the 2x2 probe unitary in the toggling frame and FFTs, giving the
              standard qubit noise susceptibility.  The 3-level Fe is smaller
              because the differential clock measurement cancels common-mode
              noise on |g>; this comparison makes that suppression explicit.

Protocol colours match across all three method types.

GPS qubit proxy: GPS drives m complete Rabi cycles on the probe transition.
In a 2-level model the |f> reference is absent, so the GPS sequence reduces
to a plain continuous Rabi drive continuous_rabi_sequence(omega_gps, T).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.optimize import minimize_scalar

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_pulse_suite.core.multilevel import (
    MultiLevelPulseSequence,
    multilevel_ramsey,
)
from quantum_pulse_suite.core.three_level_filter import (
    analytic_three_level_filter,
    fft_three_level_filter,
)
from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.global_phase_spectroscopy import (
    GlobalPhaseSpectroscopySequence,
)
from quantum_pulse_suite import (
    SIGMA_Z,
    ramsey_sequence,
    continuous_ramsey_sequence,
    continuous_rabi_sequence,
    fft_filter_function,
    noise_susceptibility_from_matrix,
)

# =============================================================================
# Configuration
# =============================================================================

T_TOTAL    = 2 * np.pi
OMEGA_FAST = 20 * np.pi

FREQ_MIN = 0.1
FREQ_MAX = 50
N_FREQ   = 2000
N_FFT    = 2048       # time samples for both FFT paths

FIGURE_SIZE = (10, 6)
DPI = 150
SAVE_FIGURE = True
OUTPUT_DIR  = 'figures/qubit_performance_plots'

M_Z_SIGMA_Y = 0.0

_OPT_RANGE = (0.001, 1.0)
_OPT_NGRID = 30
_EPS_GRAD  = 0.005


# =============================================================================
# Detuning optimisation (unchanged from original)
# =============================================================================

def _signal(build_func, delta):
    seq = build_func(delta)
    U = seq.total_subspace_unitary()
    return -float(np.imag(U[0, 0]))


def find_optimal_delta(build_func, frequencies,
                       delta_range=_OPT_RANGE, n_grid=_OPT_NGRID):
    def objective(delta):
        slope = (_signal(build_func, delta + _EPS_GRAD)
                 - _signal(build_func, delta - _EPS_GRAD)) / (2 * _EPS_GRAD)
        if abs(slope) < 1e-10:
            return np.inf
        seq = build_func(delta)
        seq.compute_polynomials()
        _, Fe, _, _ = analytic_three_level_filter(seq, frequencies, m_z=M_Z_SIGMA_Y)
        noise_int = float(np.trapezoid(Fe / frequencies, x=frequencies))
        return noise_int / slope ** 2

    deltas = np.linspace(delta_range[0], delta_range[1], n_grid)
    vals   = [objective(d) for d in deltas]

    finite = [(v, i) for i, v in enumerate(vals) if np.isfinite(v)]
    if not finite:
        return float(delta_range[0])

    i_best = min(finite)[1]
    step   = (delta_range[1] - delta_range[0]) / max(n_grid - 1, 1)
    lo     = max(delta_range[0], deltas[i_best] - step)
    hi     = min(delta_range[1], deltas[i_best] + step)

    if hi - lo < 1e-8:
        return float(deltas[i_best])

    result = minimize_scalar(objective, bounds=(lo, hi), method='bounded')
    return float(result.x)


# =============================================================================
# FFT helpers
# =============================================================================

def _fft_fe_3level(seq, m_z=M_Z_SIGMA_Y):
    """Fe via unitary propagation + FFT (3-level path)."""
    freqs, Fe, _, _ = fft_three_level_filter(seq, n_samples=N_FFT, m_z=m_z)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    return freqs[mask], Fe[mask]



def _fft_ns_2level(qubit_seq):
    """Noise susceptibility via toggling-frame FFT (2-level/qubit path)."""
    freqs, F_mat = fft_filter_function(qubit_seq, SIGMA_Z / 2, n_samples=N_FFT)
    ns = noise_susceptibility_from_matrix(F_mat)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    return freqs[mask], ns[mask]


# =============================================================================
# Plotting helper
# =============================================================================

def _add_triple(ax, color, label,
                freq_ana, Fe_ana,
                freq_3L, Fe_3L,
                freq_2L, ns_2L):
    """Plot analytic (solid), FFT-3L (dashed), FFT-2L (dotted) for one protocol."""
    kw = dict(lw=2)
    ax.loglog(freq_ana, Fe_ana / T_TOTAL**2 + 1e-20,
              color=color, ls='-', **kw, label=label)
    ax.loglog(freq_3L, Fe_3L / T_TOTAL**2 + 1e-20,
              color=color, ls='--', lw=1.4, alpha=0.85)
    ax.loglog(freq_2L, ns_2L / T_TOTAL**2 + 1e-20,
              color=color, ls=':', lw=1.4, alpha=0.85)


# =============================================================================
# Main
# =============================================================================

def main():
    fig_dir = Path(__file__).parent.parent / OUTPUT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    system  = ThreeLevelClock()

    tau_pi2  = np.pi / (2 * OMEGA_FAST)
    tau_free = T_TOTAL - 2 * tau_pi2

    # ── Instantaneous Ramsey ───────────────────────────────────────────────
    d_ri = 0.0
    print(f"Instantaneous Ramsey (delta={d_ri})")

    seq = multilevel_ramsey(system, system.probe, tau=T_TOTAL, delta=d_ri)
    seq.compute_polynomials()
    _, Fe_ana, _, _ = analytic_three_level_filter(seq, frequencies, m_z=M_Z_SIGMA_Y)

    f3L, Fe3L = _fft_fe_3level(seq)
    f2L, ns2L = _fft_ns_2level(ramsey_sequence(tau=T_TOTAL, delta=d_ri))

    _add_triple(ax, 'C0', rf'Ramsey $F_e$ (instant, $\delta={d_ri:.2f}$)',
                frequencies, Fe_ana, f3L, Fe3L, f2L, ns2L)

    # ── Continuous Ramsey ──────────────────────────────────────────────────
    d_rc = 0.0
    print(f"Continuous Ramsey (delta={d_rc})")

    seq = MultiLevelPulseSequence(system, system.probe)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], d_rc, tau_pi2)
    seq.add_free_evolution(tau_free, d_rc)
    seq.add_continuous_pulse(OMEGA_FAST, [1, 0, 0], d_rc, tau_pi2)
    seq.compute_polynomials()
    _, Fe_ana, _, _ = analytic_three_level_filter(seq, frequencies, m_z=M_Z_SIGMA_Y)

    f3L, Fe3L = _fft_fe_3level(seq)
    f2L, ns2L = _fft_ns_2level(
        continuous_ramsey_sequence(omega=OMEGA_FAST, tau=T_TOTAL, delta=d_rc))

    _add_triple(ax, 'C5', rf'Ramsey $F_e$ (continuous, $\delta={d_rc:.2f}$)',
                frequencies, Fe_ana, f3L, Fe3L, f2L, ns2L)

    # ── Rabi m=1 ──────────────────────────────────────────────────────────
    print("Optimising delta for Rabi m=1...")
    omega_rabi =  np.pi / T_TOTAL

    def _build_rabi(delta):
        s = MultiLevelPulseSequence(system, system.probe)
        s.add_continuous_pulse(omega_rabi, [1, 0, 0], delta, T_TOTAL)
        return s

    d_rabi = 0 #find_optimal_delta(_build_rabi, frequencies,
               #                 delta_range=(0.01, 2.0), n_grid=50)
    print(f"  delta_opt = {d_rabi:.4f}")

    seq = _build_rabi(d_rabi)
    seq.compute_polynomials()
    _, Fe_ana, _, _ = analytic_three_level_filter(seq, frequencies, m_z=M_Z_SIGMA_Y)

    f3L, Fe3L = _fft_fe_3level(seq)
    f2L, ns2L = _fft_ns_2level(
        continuous_rabi_sequence(omega=omega_rabi, tau=T_TOTAL, delta=d_rabi))

    _add_triple(ax, 'C1', rf'Rabi $F_e$ (m=1, $\delta={d_rabi:.2f}$)',
                frequencies, Fe_ana, f3L, Fe3L, f2L, ns2L)

    # ── GPS m=1, 2, 8 — at resonance ──────────────────────────────────────
    gps_cycles = [1, 2, 8]
    gps_colors = ['C2', 'C3', 'C4']

    for n_cyc, color in zip(gps_cycles, gps_colors):
        print(f"Computing GPS m={n_cyc} at delta=0 ...")
        omega_gps = 2 * np.pi * n_cyc / T_TOTAL

        gps = GlobalPhaseSpectroscopySequence(
            system, n_cycles=n_cyc, omega=omega_gps, delta=0.0)
        gps._sequence.compute_polynomials()
        _, Fe_ana, _, _ = analytic_three_level_filter(
            gps._sequence, frequencies, m_z=M_Z_SIGMA_Y)

        f3L, Fe3L = _fft_fe_3level(gps._sequence)
        # 2-level: GPS → continuous Rabi on probe qubit (no |f> reference)
        f2L, ns2L = _fft_ns_2level(
            continuous_rabi_sequence(omega=omega_gps, tau=T_TOTAL, delta=0.0))

        _add_triple(ax, color, rf'GPS $F_e$ (m={n_cyc}, $\delta=0$)',
                    frequencies, Fe_ana, f3L, Fe3L, f2L, ns2L)

    # ── Axes ───────────────────────────────────────────────────────────────
    ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$F_e(\omega) / T^2$', fontsize=12)
    ax.set_title(
        r'$\sigma_y$ Frequency Sensitivity ($F_e$)'
        f'\n$T = {T_TOTAL:.2f}$, optimal $\\delta$ per protocol',
        fontsize=12)
    ax.set_xlim([FREQ_MIN, FREQ_MAX])
    ax.grid(True, alpha=0.3, which='both')

    # Protocol legend (by colour, bottom-left)
    protocol_legend = ax.legend(fontsize=9, loc='lower left')

    # Method legend (by linestyle, top-right)
    method_handles = [
        Line2D([0], [0], color='k', lw=2,   ls='-',
               label='Analytic (3-level)'),
        Line2D([0], [0], color='k', lw=1.4, ls='--', alpha=0.85,
               label='FFT (3-level)'),
        Line2D([0], [0], color='k', lw=1.4, ls=':', alpha=0.85,
               label='FFT (2-level qubit)'),
    ]
    ax.legend(handles=method_handles, fontsize=9, loc='upper right')
    ax.add_artist(protocol_legend)

    plt.tight_layout()

    if SAVE_FIGURE:
        for ext in ['pdf', 'png']:
            path = fig_dir / f'sigma_y_qubit_vs_gps.{ext}'
            fig.savefig(path, dpi=DPI, bbox_inches='tight')
            print(f"Saved: {path}")

    plt.close('all')


if __name__ == '__main__':
    main()
