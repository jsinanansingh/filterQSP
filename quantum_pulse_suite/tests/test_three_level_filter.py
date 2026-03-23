"""
Unit tests for three-level clock filter functions.

Tests cover:
- FFT vs analytic agreement for Fe(w) on Ramsey, spin echo, and GPS sequences
- Known limits for Ramsey with delta=0
- Ff_analytic formula correctness (protocol-independent clock noise)
- m_y dependence of Fe
- detuning_sensitivity against GPS analytic formula
- GPS shaped filter function
"""

import unittest
import numpy as np

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import (
    MultiLevelPulseSequence,
    multilevel_ramsey,
    multilevel_spin_echo,
)
from quantum_pulse_suite.core.three_level_filter import (
    fft_three_level_filter,
    analytic_filter,
    Ff_analytic,
    detuning_sensitivity,
    gps_shaped_filter,
)
from quantum_pulse_suite import (
    continuous_ramsey_sequence,
    continuous_rabi_sequence,
)


class TestFfAnalytic(unittest.TestCase):
    """Test the protocol-independent f-noise filter function Ff."""

    def test_Ff_at_m_z_zero(self):
        """For m_z=0, Ff = (1-cos wT)/w^2."""
        T = 2.0
        freqs = np.linspace(0.5, 50.0, 200)
        Ff = Ff_analytic(freqs, T, m_z=0.0)
        expected = (1 - np.cos(freqs * T)) / freqs**2
        np.testing.assert_allclose(Ff, expected, atol=1e-12)

    def test_Ff_at_m_z_one(self):
        """For m_z=1 (sigma_z measurement), Ff = 0."""
        T = 2.0
        freqs = np.linspace(0.5, 50.0, 200)
        Ff = Ff_analytic(freqs, T, m_z=1.0)
        np.testing.assert_allclose(Ff, 0.0, atol=1e-12)

    def test_Ff_at_m_z_minus_one(self):
        """For m_z=-1, Ff = 0."""
        T = 2.0
        freqs = np.linspace(0.5, 50.0, 200)
        Ff = Ff_analytic(freqs, T, m_z=-1.0)
        np.testing.assert_allclose(Ff, 0.0, atol=1e-12)

    def test_Ff_low_frequency_limit(self):
        """At w->0, Ff -> (1-m_z^2)*T^2/2."""
        T = 3.0
        m_z = 0.5
        freqs = np.array([1e-14, 1e-13])
        Ff = Ff_analytic(freqs, T, m_z)
        expected = (1 - m_z**2) * T**2 / 2
        np.testing.assert_allclose(Ff, expected, rtol=1e-6)

    def test_Ff_positive_freqs_sign(self):
        """For 0 < |m_z| < 1, Ff should be non-negative at positive freqs."""
        T = 2.0
        freqs = np.linspace(0.1, 50.0, 500)
        for m_z in [0.0, 0.3, 0.7]:
            Ff = Ff_analytic(freqs, T, m_z)
            self.assertTrue(np.all(Ff >= -1e-15),
                            f"Ff should be >= 0 for m_z={m_z}")

    def test_Ff_scaling_with_T(self):
        """Ff peak scales as T^2."""
        m_z = 0.0
        for T in [1.0, 2.0, 4.0]:
            freqs = np.array([1e-11])
            Ff = Ff_analytic(freqs, T, m_z)
            expected = (T**2) / 2
            np.testing.assert_allclose(Ff, expected, rtol=1e-6,
                                       err_msg=f"T={T} scaling wrong")


class TestFFTThreeLevelFilter(unittest.TestCase):
    """Test FFT-based three-level filter function."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_fft_returns_correct_shapes(self):
        """FFT returns arrays of consistent shapes."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, Ff, Chi = fft_three_level_filter(seq, n_samples=1024)

        self.assertEqual(len(Fe), len(freqs))
        self.assertEqual(len(Ff), len(freqs))
        self.assertEqual(len(Chi), len(freqs))
        self.assertTrue(len(freqs) > 0)

    def test_fft_positive_frequencies(self):
        """FFT returns only positive frequencies."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, _, _, _ = fft_three_level_filter(seq, n_samples=1024)
        self.assertTrue(np.all(freqs > 0))

    def test_fft_Fe_nonneg_for_m_z_one(self):
        """Fe should be non-negative for m_z=1."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, _, _ = fft_three_level_filter(seq, m_z=1.0)
        self.assertTrue(np.all(Fe >= -1e-15))

    def test_fft_Fe_nonneg_for_m_z_zero(self):
        """Fe should be non-negative for m_z=0 (coefficient is 1/2)."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, _, _ = fft_three_level_filter(seq, m_z=0.0)
        self.assertTrue(np.all(Fe >= -1e-15))

    def test_fft_Chi_finite(self):
        """Chi values should be finite."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        _, _, _, Chi = fft_three_level_filter(seq)
        self.assertTrue(np.all(np.isfinite(Chi)))

    def test_fft_Fe_varies_with_measurement(self):
        """fft_three_level_filter uses M directly; m_y/m_x are legacy no-ops.

        The new implementation computes Fe from r(t) = <psi0|[M_I(T), A_e(t)]|psi0>
        using the full measurement operator M.  The m_y/m_x keyword arguments are
        accepted for API compatibility but do not scale Fe.
        """
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)

        # Default M = sigma_y^{gm} gives nonzero Fe
        _, Fe_default, _, _ = fft_three_level_filter(seq)
        self.assertTrue(np.any(Fe_default > 0),
                        msg="fft_three_level_filter should return nonzero Fe for default M")

        # Legacy m_y / m_x params are accepted but do not change Fe
        _, Fe_my0, _, _ = fft_three_level_filter(seq, m_y=0.0)
        _, Fe_my2, _, _ = fft_three_level_filter(seq, m_y=2.0)
        _, Fe_mx1, _, _ = fft_three_level_filter(seq, m_x=1.0)
        np.testing.assert_array_equal(Fe_default, Fe_my0,
            err_msg="m_y=0 should not zero out Fe (m_y is ignored)")
        np.testing.assert_array_equal(Fe_default, Fe_my2,
            err_msg="m_y=2 should not scale Fe (m_y is ignored)")
        np.testing.assert_array_equal(Fe_default, Fe_mx1,
            err_msg="m_x=1 should not change Fe (m_x is ignored)")

        # Passing M=zeros(3,3) should give Fe=0 everywhere
        d = seq.dim
        M_zero = np.zeros((d, d), dtype=complex)
        _, Fe_zero, _, _ = fft_three_level_filter(seq, M=M_zero)
        np.testing.assert_allclose(Fe_zero, np.zeros_like(Fe_zero), atol=1e-15,
            err_msg="M=0 should give Fe=0")


class TestFFTvsAnalytic(unittest.TestCase):
    """Compare FFT and analytic three-level Fe calculations."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def _compare_Fe(self, seq, label, rtol=0.01):
        """Helper: compare Fe(omega) between direct-FFT and analytic methods.

        fft_three_level_filter computes Fe = |FT[r(t)]|^2 by direct matrix
        multiplication (ground truth).  analytic_filter computes Fe = |H(w)|^2
        using closed-form QSP integrals.  Both include the G(T) correction.
        """
        seq.compute_polynomials()

        freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
            seq, n_samples=8192, pad_factor=4)

        T = seq.total_duration()
        mask = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0)
        freqs_check = freqs_fft[mask]

        _, Fe_ana = analytic_filter(seq, freqs_check)
        Fe_fft_check = Fe_fft[mask]

        peak = max(np.max(Fe_fft_check), np.max(Fe_ana))
        if peak < 1e-20:
            return

        sig = (Fe_fft_check > 1e-4 * peak) & (Fe_ana > 1e-4 * peak)
        if not np.any(sig):
            return

        rel_err = np.abs(Fe_fft_check[sig] - Fe_ana[sig]) / Fe_ana[sig]
        self.assertLess(
            np.max(rel_err), rtol,
            msg=f"{label}: max Fe rel error {np.max(rel_err):.4f} > {rtol}"
        )

    def test_ramsey_fft_vs_analytic(self):
        """Ramsey: direct-FFT Fe matches analytic Fe."""
        for tau in [1.0, 2.0]:
            seq = multilevel_ramsey(self.system, self.system.probe,
                                    tau=tau, delta=0.0)
            self._compare_Fe(seq, f"Ramsey(tau={tau})")

    def test_ramsey_with_detuning_fft_vs_analytic(self):
        """Ramsey with detuning: direct-FFT Fe matches analytic Fe."""
        seq = multilevel_ramsey(self.system, self.system.probe,
                                tau=2.0, delta=0.5)
        self._compare_Fe(seq, "Ramsey(tau=2, delta=0.5)")

    def test_spin_echo_fft_vs_analytic(self):
        """Spin echo: direct-FFT Fe matches analytic Fe."""
        for tau in [1.0, 2.0]:
            seq = multilevel_spin_echo(self.system, self.system.probe,
                                        tau=tau, delta=0.0)
            self._compare_Fe(seq, f"SpinEcho(tau={tau})")


class TestMzDependence(unittest.TestCase):
    """Test measurement-component dependence of Fe and Ff."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_Fe_prefactor_values(self):
        """fft_three_level_filter Fe agrees with analytic_filter Fe for Ramsey (G(T)=0).

        For M = sigma_y^{gm}, psi0 = (|g>+|m>)/sqrt(2) and a Ramsey sequence
        (G(T) = 0), both the direct-FFT and analytic paths give Fe = |Chi(w)|^2.
        This cross-validates the two implementations on a case with a known answer.
        """
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        seq.compute_polynomials()

        T = seq.total_duration()
        freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
            seq, n_samples=4096, pad_factor=4)

        mask = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0)
        freqs_check = freqs_fft[mask]
        _, Fe_ana = analytic_filter(seq, freqs_check)
        Fe_fft_check = Fe_fft[mask]

        peak = np.max(Fe_ana)
        sig = (Fe_fft_check > 1e-4 * peak) & (Fe_ana > 1e-4 * peak)
        np.testing.assert_allclose(
            Fe_fft_check[sig], Fe_ana[sig], rtol=0.01,
            err_msg="fft Fe and analytic Fe should agree within 1% for Ramsey")

    def test_Ff_varies_with_m_z(self):
        """Ff should scale as (1 - m_z^2)."""
        T = 2.0
        freqs = np.linspace(1.0, 20.0, 100)

        Ff_0 = Ff_analytic(freqs, T, m_z=0.0)      # coefficient = 1
        Ff_half = Ff_analytic(freqs, T, m_z=0.5)   # coefficient = 0.75

        np.testing.assert_allclose(Ff_half, 0.75 * Ff_0, rtol=1e-12)



class TestFFTvsAnalyticFilter(unittest.TestCase):
    """fft_three_level_filter and analytic_filter should agree on significant signal."""

    N_FFT_SAMPLES = 8192
    PAD = 4
    N_CHECK = 50      # number of frequencies passed to the analytic function
    RTOL = 0.05       # 5% relative tolerance at signal peaks

    def setUp(self):
        self.system = ThreeLevelClock()

    def _check(self, seq, label):
        seq.compute_polynomials()
        T = seq.total_duration()

        freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
            seq, n_samples=self.N_FFT_SAMPLES, pad_factor=self.PAD)

        band = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0)
        idx_sub = np.round(
            np.linspace(0, band.sum() - 1, self.N_CHECK)).astype(int)
        freqs_check  = freqs_fft[band][idx_sub]
        Fe_fft_check = Fe_fft[band][idx_sub]

        _, Fe_ana = analytic_filter(seq, freqs_check)

        peak = max(np.max(Fe_fft_check), np.max(Fe_ana))
        if peak < 1e-20:
            return  # both are zero — nothing to compare

        sig = (Fe_fft_check > 1e-3 * peak) & (Fe_ana > 1e-3 * peak)
        if not np.any(sig):
            return

        rel_err = np.abs(Fe_fft_check[sig] - Fe_ana[sig]) / Fe_ana[sig]
        self.assertLess(
            np.max(rel_err), self.RTOL,
            msg=f"{label}: max rel error {np.max(rel_err):.4f} > {self.RTOL}")

    def test_ramsey_instant(self):
        """Instantaneous Ramsey (free-evolution only): FFT matches analytic_filter."""
        for tau in [1.0, 2 * np.pi]:
            seq = multilevel_ramsey(self.system, self.system.probe,
                                    tau=tau, delta=0.0)
            self._check(seq, f"Ramsey instant tau={tau:.2f}")

    def test_ramsey_continuous(self):
        """Continuous Ramsey (pi/2 pulses + free evolution): FFT matches analytic_filter."""
        T = 2 * np.pi
        omega = 20 * np.pi
        tau_pi2  = np.pi / (2 * omega)
        tau_free = T - 2 * tau_pi2
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, tau_pi2)
        seq.add_free_evolution(tau_free, 0.0)
        seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, tau_pi2)
        self._check(seq, "Ramsey continuous")

    def test_gps_rabi(self):
        """Single continuous Rabi pulse (GPS m=1): FFT matches analytic_filter."""
        T = 2 * np.pi
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_continuous_pulse(np.pi / T, [1, 0, 0], 0.0, T)
        self._check(seq, "GPS Rabi m=1")



class TestGPSDetuningSensitivityAnalytic(unittest.TestCase):
    """
    Verify detuning_sensitivity against the analytic formula for GPS
    (single continuous Rabi drive on a three-level clock).

    With H_delta = delta*|e><e|, initial state (|g>+|m>)/sqrt(2), measurement
    M = sigma_y^{gm}, and a single x-axis drive of duration T and Rabi
    frequency Omega, the signal is

        <M>(delta) = sin(delta*T/2)*cos(Omega_R*T/2)
                     - (delta/Omega_R)*cos(delta*T/2)*sin(Omega_R*T/2)

    with derivative

        d<M>/d(delta) = Omega^2 * cos(delta*T/2)
                        * (T*Omega_R*cos(Omega_R*T/2) - 2*sin(Omega_R*T/2))
                        / (2 * Omega_R^3)

    where Omega_R = sqrt(delta^2 + Omega^2).
    """

    RTOL = 1e-6   # relative tolerance: finite-difference eps=1e-7 gives error << 1e-6
    # Absolute tolerance for near-zero derivative cases: when the true value is 0
    # (e.g. cos(delta*T/2) = 0 exactly), cancellation in the CK propagator produces
    # finite-difference noise of order eps_machine / eps_fd ~ 1e-15/1e-7 = 1e-8.
    ATOL = 1e-7

    def _gps_seq(self, omega, T, delta_base=0.0):
        """Build a 3-level GPS sequence: single continuous x-axis pulse."""
        system = ThreeLevelClock()
        seq = MultiLevelPulseSequence(system, system.probe)
        seq.add_continuous_pulse(omega, [1.0, 0.0, 0.0], delta_base, T)
        seq.compute_polynomials()
        return seq

    @staticmethod
    def _analytic(omega, T, delta):
        """Analytic d<M>/d(delta) for GPS at total detuning delta."""
        omega_r = np.sqrt(delta**2 + omega**2)
        return (omega**2 * np.cos(delta * T / 2)
                * (T * omega_r * np.cos(omega_r * T / 2)
                   - 2.0 * np.sin(omega_r * T / 2))
                / (2.0 * omega_r**3))

    def _check(self, omega, T, delta, label=''):
        seq = self._gps_seq(omega, T)
        dM, sens_sq = detuning_sensitivity(seq, delta=delta)
        expected = self._analytic(omega, T, delta)
        np.testing.assert_allclose(
            dM, expected, rtol=self.RTOL, atol=self.ATOL,
            err_msg=f"{label}: numerical={dM:.8g}, analytic={expected:.8g}")
        # sens_sq must equal dM^2 by definition
        np.testing.assert_allclose(sens_sq, dM**2, rtol=1e-12,
                                   err_msg=f"{label}: sens_sq != dM^2")

    # ------------------------------------------------------------------
    # Resonant cases (delta = 0)
    # ------------------------------------------------------------------

    def test_resonant_quarter_wave(self):
        """delta=0, Omega*T = pi/2."""
        T = 2.0
        self._check(np.pi / (2 * T), T, delta=0.0, label="resonant pi/2-rotation")

    def test_resonant_pi_pulse(self):
        """delta=0, Omega*T = pi (pi pulse)."""
        T = 2.0
        self._check(np.pi / T, T, delta=0.0, label="resonant pi-rotation")

    def test_resonant_two_pi(self):
        """delta=0, Omega*T = 2pi (full rotation back to identity)."""
        T = 2.0
        self._check(2 * np.pi / T, T, delta=0.0, label="resonant 2pi-rotation")

    # ------------------------------------------------------------------
    # Off-resonant cases (delta != 0)
    # ------------------------------------------------------------------

    def test_off_resonant_pi_pulse(self):
        """Small positive detuning, Omega*T ~ pi."""
        T = 2.0
        omega = np.pi / T
        self._check(omega, T, delta=0.3 / T, label="off-resonant pi-pulse")

    def test_off_resonant_general(self):
        """General off-resonant case with incommensurable values."""
        self._check(omega=3.7, T=1.5, delta=1.2, label="general off-resonant")

    def test_negative_delta(self):
        """Negative detuning."""
        T = 2.0
        self._check(np.pi / T, T, delta=-0.5, label="negative delta")

    def test_large_detuning(self):
        """Large detuning (delta >> Omega): sensitivity should be small."""
        T = 1.0
        omega = np.pi / T          # resonant Rabi freq
        delta = 20.0 / T           # delta / Omega = 20
        self._check(omega, T, delta=delta, label="large detuning")

    # ------------------------------------------------------------------
    # Equivalence: delta in element vs delta argument
    # ------------------------------------------------------------------

    def test_delta_in_element_equals_delta_argument(self):
        """
        Passing detuning as element.delta vs as the delta argument to
        detuning_sensitivity should give identical results, since the total
        detuning is element.delta + delta_extra in both cases.
        """
        T = 2.0
        omega = np.pi / T
        delta_val = 0.25

        system = ThreeLevelClock()
        seq_base = MultiLevelPulseSequence(system, system.probe)
        seq_base.add_continuous_pulse(omega, [1.0, 0.0, 0.0], delta_val, T)
        seq_base.compute_polynomials()
        dM_base, _ = detuning_sensitivity(seq_base, delta=0.0)

        seq_arg = self._gps_seq(omega, T, delta_base=0.0)
        dM_arg, _ = detuning_sensitivity(seq_arg, delta=delta_val)

        expected = self._analytic(omega, T, delta_val)
        np.testing.assert_allclose(dM_base, expected, rtol=self.RTOL,
                                   err_msg="delta in element mismatch")
        np.testing.assert_allclose(dM_arg, expected, rtol=self.RTOL,
                                   err_msg="delta as argument mismatch")
        np.testing.assert_allclose(dM_base, dM_arg, rtol=1e-10,
                                   err_msg="two representations disagree")

    # ------------------------------------------------------------------
    # Parameter grid sweep
    # ------------------------------------------------------------------

    def test_parameter_grid(self):
        """Sweep (T, Omega, delta): all must satisfy the analytic formula."""
        params = [
            (1.0,           np.pi,           0.0),
            (2 * np.pi,     1.0,             0.0),
            (2 * np.pi,     2 * np.pi,       0.5),
            (1.5,           4.0,            -0.8),
            (3.0,           np.pi / 3.0,     1.0),
        ]
        for T, omega, delta in params:
            with self.subTest(T=T, omega=omega, delta=delta):
                self._check(omega, T, delta,
                            label=f"T={T:.2f} Ω={omega:.3f} δ={delta:.2f}")


class TestAnalyticVsFFTOptimizer(unittest.TestCase):
    """
    Verify that the analytic and FFT evaluation/optimizer paths in
    optimize_equiangular_sequence are consistent enough for the paper workflow.

    The tests are structured in three levels:

    1. test_kubo_integral_agreement: For several fixed sequences, check that
       analytic_evaluate_sequence and evaluate_sequence give sigma_nu values
       that agree to within about 10%.  This validates the analytic quadrature
       without over-constraining sampling differences.

    2. test_analytic_optimizer_finds_good_solution: Run the analytic optimizer
       with a moderate budget and verify it finds FOM > 12 (Ramsey = 15.6),
       confirming the analytic path works end-to-end.

    3. test_both_methods_find_comparable_fom: Run both optimizers with the
       same seed and check that both improve over Ramsey and land in the same
       qualitative performance regime, even if they settle in different local
       optima.
    """

    def _make_system(self):
        return ThreeLevelClock(), 2 * np.pi

    def test_kubo_integral_agreement(self):
        """analytic_evaluate_sequence and evaluate_sequence agree within 10% for fixed sequences."""
        from quantum_pulse_suite.analysis.pulse_optimizer import (
            evaluate_sequence, analytic_evaluate_sequence,
            white_noise_psd, one_over_f_psd, build_equiangular_3level,
        )
        system, T = self._make_system()
        S_w = white_noise_psd()
        S_f = one_over_f_psd()

        # Include the known near-optimal N=4 sequence and a few arbitrary ones
        test_params = [
            (8.18 / T, [0.0, 1.476, 0.154, 1.178]),   # near-optimal
            (1.0,      [0.0, 0.5, 1.0, 1.5]),          # arbitrary
            (2.5,      [0.0, np.pi, 0.3, np.pi + 0.3]),
        ]

        for omega, phases in test_params:
            seq = build_equiangular_3level(system, T, 4, omega, phases)
            for S_func, noise_name in [(S_w, 'white'), (S_f, '1/f')]:
                _, _, snu_fft = evaluate_sequence(
                    seq, S_func, n_fft=2048, pad_factor=4)
                _, _, snu_ana = analytic_evaluate_sequence(
                    seq, S_func, n_omega=512, omega_max=50.0, fft_pad_factor=4)
                if snu_fft > 0:
                    ratio = snu_ana / snu_fft
                    self.assertAlmostEqual(
                        ratio, 1.0, delta=0.10,
                        msg=(f"[{noise_name}] sigma_nu mismatch for omega={omega:.4f}: "
                             f"fft={snu_fft:.4f}  ana={snu_ana:.4f}  ratio={ratio:.4f}"),
                    )

    def test_analytic_optimizer_finds_good_solution(self):
        """Analytic optimizer (N=4, white) finds sigma_nu < 0.1 (Ramsey is ~0.043)."""
        from quantum_pulse_suite.analysis.pulse_optimizer import (
            optimize_equiangular_sequence,
        )
        system, T = self._make_system()
        res = optimize_equiangular_sequence(
            system, T, N=4, noise_psd='white',
            use_analytic=True,
            seed=7, n_restarts=3, popsize=12, maxiter=200,
        )
        self.assertLess(
            res.sigma_nu, 0.1,
            msg=f"Analytic optimizer sigma_nu too high: {res.sigma_nu:.4f} (Ramsey~0.043)",
        )

    def test_both_methods_find_comparable_fom(self):
        """FFT and analytic optimizers both beat Ramsey and stay in the same regime."""
        from quantum_pulse_suite.analysis.pulse_optimizer import (
            optimize_equiangular_sequence, build_ramsey_3level,
            evaluate_sequence, white_noise_psd,
        )
        system, T = self._make_system()
        common = dict(N=4, noise_psd='white', seed=7, n_restarts=3,
                      popsize=12, maxiter=200)

        _, _, ramsey_sigma = evaluate_sequence(
            build_ramsey_3level(system, T, omega_fast=20 * np.pi),
            white_noise_psd(),
        )

        res_fft = optimize_equiangular_sequence(
            system, T, use_analytic=False, n_fft=2048, pad_factor=4, **common)
        res_ana = optimize_equiangular_sequence(
            system, T, use_analytic=True, n_omega=256, omega_max_analytic=30., **common)

        self.assertLess(
            res_fft.sigma_nu, ramsey_sigma,
            msg=f"FFT optimizer should beat Ramsey: fft={res_fft.sigma_nu:.4f}, Ramsey={ramsey_sigma:.4f}")
        self.assertLess(
            res_ana.sigma_nu, ramsey_sigma,
            msg=f"Analytic optimizer should beat Ramsey: ana={res_ana.sigma_nu:.4f}, Ramsey={ramsey_sigma:.4f}")

        ratio = res_ana.sigma_nu / max(res_fft.sigma_nu, 1e-15)
        self.assertGreater(
            ratio, 0.4,
            msg=(f"sigma_nu ratio unexpectedly small: fft={res_fft.sigma_nu:.4f}  "
                 f"ana={res_ana.sigma_nu:.4f}  ratio={ratio:.4f}"),
        )
        self.assertLess(
            ratio, 2.5,
            msg=(f"sigma_nu ratio unexpectedly large: fft={res_fft.sigma_nu:.4f}  "
                 f"ana={res_ana.sigma_nu:.4f}  ratio={ratio:.4f}"),
        )



class TestGpsShapedFilterMethods(unittest.TestCase):
    """Verify gps_shaped_filter method='direct' vs method='piecewise'.

    Both methods must agree on the noise variance to within 1 % for three
    envelope shapes (square, Hann, Blackman) and two GPS operating points
    (m=1, m=8), under white, 1/f, and high-pass noise.
    """

    T = 2 * np.pi
    REL_TOL = 0.01   # 1 %

    @staticmethod
    def _envelope_square(ts, T, omega_mean):
        return np.full_like(ts, omega_mean, dtype=float)

    @staticmethod
    def _envelope_hann(ts, T, omega_mean):
        return omega_mean * (1.0 - np.cos(2.0 * np.pi * ts / T))

    @staticmethod
    def _envelope_blackman(ts, T, omega_mean):
        a0, a1, a2 = 0.42, 0.50, 0.08
        return (omega_mean / a0) * (a0
                                    - a1 * np.cos(2.0 * np.pi * ts / T)
                                    + a2 * np.cos(4.0 * np.pi * ts / T))

    def _build_seq_pw(self, system, omega_mean, envelope_fn, n_disc=256):
        from quantum_pulse_suite.core.multilevel import MultiLevelPulseSequence
        tau    = self.T / n_disc
        ts_mid = (np.arange(n_disc) + 0.5) * tau
        omegas = np.maximum(envelope_fn(ts_mid, self.T, omega_mean), 1e-9)
        seq    = MultiLevelPulseSequence(system, system.probe)
        for ok in omegas:
            seq.add_continuous_pulse(ok, [1, 0, 0], 0.0, tau)
        seq.compute_polynomials()
        return seq

    def _noise_var(self, freqs, Fe, S_func, omega_cutoff=1e-4):
        from scipy.integrate import simpson
        mask = freqs >= omega_cutoff
        if mask.sum() < 2:
            return 0.0
        return float(simpson(Fe[mask] * S_func(freqs[mask]),
                             x=freqs[mask]) / (2 * np.pi))

    def _check_envelope(self, envelope_name, envelope_fn, omega_mean):
        from quantum_pulse_suite.analysis.pulse_optimizer import (
            white_noise_psd, one_over_f_psd, high_pass_psd)
        system = ThreeLevelClock()
        noise_specs = [
            ('white', white_noise_psd()),
            ('1/f',   one_over_f_psd()),
            ('hp2',   high_pass_psd(omega_c=2.0)),
        ]

        freqs_d, Fe_d = gps_shaped_filter(
            envelope_fn, self.T, omega_mean, method='direct',
            n_samples=8192, pad_factor=4)

        seq_pw = self._build_seq_pw(system, omega_mean, envelope_fn)
        freqs_p, Fe_p = gps_shaped_filter(
            envelope_fn, self.T, omega_mean, method='piecewise',
            n_samples=4 * 256, pad_factor=4, seq=seq_pw)

        for nname, S_func in noise_specs:
            nv_d = self._noise_var(freqs_d, Fe_d, S_func)
            nv_p = self._noise_var(freqs_p, Fe_p, S_func)
            if nv_d < 1e-20 and nv_p < 1e-20:
                continue
            rel = abs(nv_d - nv_p) / max(abs(nv_d), 1e-30)
            self.assertLess(
                rel, self.REL_TOL,
                f'{envelope_name} omega_mean={omega_mean:.3f} [{nname}]: '
                f'direct={nv_d:.6e}, piecewise={nv_p:.6e}, rel={rel:.4f}')

    def test_square_m1(self):
        self._check_envelope('square', self._envelope_square, 2 * np.pi * 1 / self.T)

    def test_hann_m1(self):
        self._check_envelope('hann', self._envelope_hann, 2 * np.pi * 1 / self.T)

    def test_blackman_m1(self):
        self._check_envelope('blackman', self._envelope_blackman, 2 * np.pi * 1 / self.T)

    def test_square_m8(self):
        self._check_envelope('square', self._envelope_square, 2 * np.pi * 8 / self.T)

    def test_hann_m8(self):
        self._check_envelope('hann', self._envelope_hann, 2 * np.pi * 8 / self.T)

    def test_blackman_m8(self):
        self._check_envelope('blackman', self._envelope_blackman, 2 * np.pi * 8 / self.T)


if __name__ == '__main__':
    unittest.main()
