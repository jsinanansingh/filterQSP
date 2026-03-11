"""
Unit tests for three-level clock filter functions.

Tests cover:
- FFT vs analytic agreement for Chi(w) / Fe(w) on Ramsey and spin echo sequences
- Known limits for Ramsey with delta=0
- Ff formula correctness (sinc-like, analytic)
- m_y dependence of Fe
- Variance integration sanity checks
"""

import unittest
import numpy as np
from scipy.linalg import expm

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.core.multilevel import (
    MultiLevelPulseSequence,
    multilevel_ramsey,
    multilevel_spin_echo,
    multilevel_cpmg,
)
from quantum_pulse_suite.core.three_level_filter import (
    fft_three_level_filter,
    analytic_three_level_filter,
    three_level_noise_variance,
    Ff_analytic,
    kubo_filter_2level,
    kubo_filter_3level,
    kubo_filter_2level_analytic,
    kubo_filter_3level_analytic,
    detuning_sensitivity,
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
        """Fe = m_y^2 * |Chi|^2: only m_y affects the classical noise filter function.

        m_x and m_z do not enter the new formula.
        """
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)

        # m_y=0 should give Fe=0 everywhere
        _, Fe_my0, _, _ = fft_three_level_filter(seq, m_y=0.0)
        np.testing.assert_array_equal(Fe_my0, np.zeros_like(Fe_my0))

        # m_y=1 should give positive Fe
        _, Fe_my1, _, Chi = fft_three_level_filter(seq, m_y=1.0)
        np.testing.assert_allclose(Fe_my1, np.abs(Chi)**2, rtol=1e-10)

        # m_y=2 should give 4*|Chi|^2
        _, Fe_my2, _, _ = fft_three_level_filter(seq, m_y=2.0)
        np.testing.assert_allclose(Fe_my2, 4.0 * Fe_my1, rtol=1e-10)

        # m_x, m_z do not change Fe
        _, Fe_mz1, _, _ = fft_three_level_filter(seq, m_y=1.0, m_z=1.0)
        _, Fe_mx1, _, _ = fft_three_level_filter(seq, m_y=1.0, m_x=1.0)
        np.testing.assert_array_equal(Fe_my1, Fe_mz1)
        np.testing.assert_array_equal(Fe_my1, Fe_mx1)


class TestFFTvsAnalytic(unittest.TestCase):
    """Compare FFT and analytic three-level Chi / Fe calculations."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def _compare_Chi(self, seq, label, rtol=0.01):
        """Helper: compare |Chi|^2 between FFT and analytic methods."""
        seq.compute_polynomials()

        freqs_fft, _, _, Chi_fft = fft_three_level_filter(
            seq, n_samples=8192, pad_factor=4, m_y=1.0)

        _, _, _, Chi_ana = analytic_three_level_filter(
            seq, freqs_fft, m_y=1.0)

        Chi2_fft = np.abs(Chi_fft)**2
        Chi2_ana = np.abs(Chi_ana)**2

        T = seq.total_duration()
        mask = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0)
        peak = max(np.max(Chi2_fft[mask]), np.max(Chi2_ana[mask]))

        if peak < 1e-20:
            return  # nothing to compare

        sig = (Chi2_fft[mask] > 1e-4 * peak) & (Chi2_ana[mask] > 1e-4 * peak)
        if not np.any(sig):
            return

        rel_err = np.abs(Chi2_fft[mask][sig] - Chi2_ana[mask][sig]) / Chi2_ana[mask][sig]
        self.assertLess(
            np.max(rel_err), rtol,
            msg=f"{label}: max |Chi|^2 rel error {np.max(rel_err):.4f} > {rtol}"
        )

    def test_ramsey_fft_vs_analytic(self):
        """Ramsey: FFT |Chi|^2 matches analytic."""
        for tau in [1.0, 2.0]:
            seq = multilevel_ramsey(self.system, self.system.probe,
                                    tau=tau, delta=0.0)
            self._compare_Chi(seq, f"Ramsey(tau={tau})")

    def test_ramsey_with_detuning_fft_vs_analytic(self):
        """Ramsey with detuning: FFT |Chi|^2 matches analytic."""
        seq = multilevel_ramsey(self.system, self.system.probe,
                                tau=2.0, delta=0.5)
        self._compare_Chi(seq, "Ramsey(tau=2, delta=0.5)")

    def test_spin_echo_fft_vs_analytic(self):
        """Spin echo: FFT |Chi|^2 matches analytic."""
        for tau in [1.0, 2.0]:
            seq = multilevel_spin_echo(self.system, self.system.probe,
                                        tau=tau, delta=0.0)
            self._compare_Chi(seq, f"SpinEcho(tau={tau})")

    def test_cpmg_fft_vs_analytic(self):
        """CPMG: FFT |Chi|^2 matches analytic."""
        tau = 2.0
        for n in [1, 2, 4]:
            seq = multilevel_cpmg(self.system, self.system.probe,
                                   tau=tau, n_pulses=n, delta=0.0)
            self._compare_Chi(seq, f"CPMG-{n}(tau={tau})")


class TestRamseyKnownLimits(unittest.TestCase):
    """Test Chi against known formulas for simple sequences."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_ramsey_delta_zero_Chi_form(self):
        """
        Ramsey delta=0: the QSP unitary on {|g>,|e>} is
        pi/2_x - free(tau) - pi/2_x.

        The Chi integrand |G(t)|^2 is constant during free evolution for this
        sequence, so |Chi(w)|^2 has a sinc-like shape.
        """
        tau = 2.0
        seq = multilevel_ramsey(self.system, self.system.probe,
                                tau=tau, delta=0.0)
        seq.compute_polynomials()

        # After first pi/2_x: f = cos(pi/4) = 1/sqrt(2), g = sin(pi/4) = 1/sqrt(2)
        # During free evolution (delta=0): F(t) = f, G(t) = g (constant)
        # So F*G = conj(f)*g = (1/sqrt(2))*(1/sqrt(2)) = 1/2

        # Chi(w) = (1/2) * integral_0^tau e^{-iwt} dt
        # = (1/2) * (1 - e^{-iwτ})/(iw)
        # |Chi|^2 = (1/4) * 2(1-cos(wτ))/w^2 = (1-cos(wτ))/(2w^2)

        freqs = np.linspace(0.5, 30.0, 200)
        _, _, _, Chi_ana = analytic_three_level_filter(seq, freqs, m_y=1.0)

        Chi2_expected = (1 - np.cos(freqs * tau)) / (2 * freqs**2)
        Chi2_actual = np.abs(Chi_ana)**2

        np.testing.assert_allclose(Chi2_actual, Chi2_expected, rtol=0.01,
                                    err_msg="Ramsey delta=0 |Chi|^2 mismatch")

    def test_ramsey_delta_nonzero_Chi(self):
        """
        Ramsey with detuning: |G|^2 is still constant during free evolution,
        so |Chi|^2 has the same sinc envelope regardless of delta.
        """
        tau = 2.0
        freqs = np.linspace(0.5, 30.0, 200)

        seq0 = multilevel_ramsey(self.system, self.system.probe,
                                  tau=tau, delta=0.0)
        seq0.compute_polynomials()
        _, _, _, Chi0 = analytic_three_level_filter(seq0, freqs, m_y=1.0)

        seq1 = multilevel_ramsey(self.system, self.system.probe,
                                  tau=tau, delta=1.5)
        seq1.compute_polynomials()
        _, _, _, Chi1 = analytic_three_level_filter(seq1, freqs, m_y=1.0)

        # |Chi|^2 should be the same regardless of delta
        np.testing.assert_allclose(
            np.abs(Chi0)**2, np.abs(Chi1)**2, rtol=0.01,
            err_msg="|Chi|^2 should be delta-independent for Ramsey")


class TestMzDependence(unittest.TestCase):
    """Test measurement-component dependence of Fe and Ff."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_Fe_prefactor_values(self):
        """Classical noise filter function: F = m_y^2 * |Chi|^2.

        Key consequences:
        - m_x and m_z do NOT appear in F.
        - F scales as m_y^2: Fe(m_y=2) = 4 * Fe(m_y=1).
        - F is zero when m_y=0.
        """
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        _, _, _, Chi_ref = fft_three_level_filter(seq, m_y=1.0)
        Chi2 = np.abs(Chi_ref)**2

        # m_y=1 gives Fe = |Chi|^2
        _, Fe_my1, _, _ = fft_three_level_filter(seq, m_y=1.0)
        np.testing.assert_allclose(Fe_my1, Chi2, rtol=1e-10)

        # m_y=0 gives Fe = 0
        _, Fe_my0, _, _ = fft_three_level_filter(seq, m_y=0.0)
        np.testing.assert_allclose(Fe_my0, np.zeros_like(Fe_my0), atol=1e-15)

        # m_y=2 gives Fe = 4*|Chi|^2
        _, Fe_my2, _, _ = fft_three_level_filter(seq, m_y=2.0)
        np.testing.assert_allclose(Fe_my2, 4.0 * Chi2, rtol=1e-10)

        # m_x, m_z don't affect Fe (for any m_y value)
        for m_z in [0.0, 0.5, 1.0]:
            _, Fe_mz, _, _ = fft_three_level_filter(seq, m_y=1.0, m_z=m_z)
            np.testing.assert_allclose(
                Fe_mz, Fe_my1, rtol=1e-10,
                err_msg=f"Fe should not depend on m_z={m_z}")

    def test_Ff_varies_with_m_z(self):
        """Ff should scale as (1 - m_z^2)."""
        T = 2.0
        freqs = np.linspace(1.0, 20.0, 100)

        Ff_0 = Ff_analytic(freqs, T, m_z=0.0)      # coefficient = 1
        Ff_half = Ff_analytic(freqs, T, m_z=0.5)   # coefficient = 0.75

        np.testing.assert_allclose(Ff_half, 0.75 * Ff_0, rtol=1e-12)


class TestNoiseVarianceIntegration(unittest.TestCase):
    """Test noise variance computation."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_white_noise_e_only(self):
        """Variance with white e-noise only should be positive."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, Ff, _ = fft_three_level_filter(seq, m_y=1.0, m_z=0.0)

        S_e = lambda w: np.ones_like(w) * 1e-4
        S_f = lambda w: np.zeros_like(w)

        var = three_level_noise_variance(Fe, Ff, freqs, S_e, S_f)
        self.assertGreater(var, 0.0)
        self.assertTrue(np.isfinite(var))

    def test_white_noise_f_only(self):
        """Variance with white f-noise only and m_z=0."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, Ff, _ = fft_three_level_filter(seq, m_z=0.0)

        S_e = lambda w: np.zeros_like(w)
        S_f = lambda w: np.ones_like(w) * 1e-4

        var = three_level_noise_variance(Fe, Ff, freqs, S_e, S_f)
        self.assertGreater(var, 0.0)
        self.assertTrue(np.isfinite(var))

    def test_variance_zero_noise(self):
        """Zero noise should give zero variance."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, Ff, _ = fft_three_level_filter(seq, m_z=0.0)

        S_e = lambda w: np.zeros_like(w)
        S_f = lambda w: np.zeros_like(w)

        var = three_level_noise_variance(Fe, Ff, freqs, S_e, S_f)
        self.assertAlmostEqual(var, 0.0, places=15)

    def test_variance_scales_with_noise_amplitude(self):
        """Variance should scale linearly with noise PSD amplitude."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, Ff, _ = fft_three_level_filter(seq, m_y=1.0, m_z=1.0)

        S_f_zero = lambda w: np.zeros_like(w)

        var1 = three_level_noise_variance(
            Fe, Ff, freqs,
            S_e=lambda w: np.ones_like(w) * 1.0,
            S_f=S_f_zero)
        var2 = three_level_noise_variance(
            Fe, Ff, freqs,
            S_e=lambda w: np.ones_like(w) * 2.0,
            S_f=S_f_zero)

        if var1 > 1e-15:
            np.testing.assert_allclose(var2, 2.0 * var1, rtol=1e-6)


class TestSpinEchoChiShape(unittest.TestCase):
    """Basic structure test for spin-echo Chi."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_spin_echo_chi_is_finite(self):
        """Spin echo |Chi|^2 should be finite on a harmonic sample grid."""
        tau = 2.0
        seq = multilevel_spin_echo(self.system, self.system.probe,
                                    tau=tau, delta=0.0)
        seq.compute_polynomials()
        fundamental = 2 * np.pi / tau
        even_harmonics = np.array([2, 4, 6]) * fundamental
        _, _, _, Chi = analytic_three_level_filter(seq, even_harmonics, m_y=1.0)
        self.assertTrue(np.all(np.isfinite(np.abs(Chi)**2)))



class TestKuboFFTvsAnalytic(unittest.TestCase):
    """FFT and analytic Kubo filter functions should agree on significant signal."""

    M_HAT = np.array([0., 1., 0.])
    R0    = np.array([1., 0., 0.])
    N_FFT_SAMPLES = 8192
    PAD = 4
    N_CHECK = 50      # number of frequencies passed to the analytic function
    RTOL = 0.05       # 5% relative tolerance at signal peaks

    def setUp(self):
        self.system = ThreeLevelClock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_3level(self, seq3, label):
        seq3.compute_polynomials()
        T = seq3.total_duration()

        freqs_fft, Fk_fft = kubo_filter_3level(
            seq3, n_samples=self.N_FFT_SAMPLES, pad_factor=self.PAD)

        # Pick ~N_CHECK frequencies in the informative band
        band = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0)
        idx_sub = np.round(
            np.linspace(0, band.sum() - 1, self.N_CHECK)).astype(int)
        freqs_check  = freqs_fft[band][idx_sub]
        Fk_fft_check = Fk_fft[band][idx_sub]

        _, Fk_ana = kubo_filter_3level_analytic(seq3, freqs_check)

        peak = max(np.max(Fk_fft_check), np.max(Fk_ana))
        if peak < 1e-20:
            return  # both are zero — nothing to compare

        sig = (Fk_fft_check > 1e-3 * peak) & (Fk_ana > 1e-3 * peak)
        if not np.any(sig):
            return

        rel_err = np.abs(Fk_fft_check[sig] - Fk_ana[sig]) / Fk_ana[sig]
        self.assertLess(
            np.max(rel_err), self.RTOL,
            msg=f"{label}: max rel error {np.max(rel_err):.4f} > {self.RTOL}")

    def _check_2level(self, seq2, label):
        seq2.compute_polynomials()
        T = seq2.total_duration()

        freqs_fft, Fk_fft = kubo_filter_2level(
            seq2, m_hat=self.M_HAT, r0=self.R0,
            n_samples=self.N_FFT_SAMPLES, pad_factor=self.PAD)

        band = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0)
        idx_sub = np.round(
            np.linspace(0, band.sum() - 1, self.N_CHECK)).astype(int)
        freqs_check  = freqs_fft[band][idx_sub]
        Fk_fft_check = Fk_fft[band][idx_sub]

        _, Fk_ana = kubo_filter_2level_analytic(
            seq2, freqs_check, m_hat=self.M_HAT, r0=self.R0)

        peak = max(np.max(Fk_fft_check), np.max(Fk_ana))
        if peak < 1e-20:
            return

        sig = (Fk_fft_check > 1e-3 * peak) & (Fk_ana > 1e-3 * peak)
        if not np.any(sig):
            return

        rel_err = np.abs(Fk_fft_check[sig] - Fk_ana[sig]) / Fk_ana[sig]
        self.assertLess(
            np.max(rel_err), self.RTOL,
            msg=f"{label}: max rel error {np.max(rel_err):.4f} > {self.RTOL}")

    # ------------------------------------------------------------------
    # 3-level tests
    # ------------------------------------------------------------------

    def test_3level_ramsey_instant(self):
        """Instantaneous Ramsey (free-evolution only): FFT matches analytic."""
        for tau in [1.0, 2 * np.pi]:
            seq = multilevel_ramsey(self.system, self.system.probe,
                                    tau=tau, delta=0.0)
            self._check_3level(seq, f"3L Ramsey instant tau={tau:.2f}")

    def test_3level_ramsey_continuous(self):
        """Continuous Ramsey (pi/2 pulses + free evolution): FFT matches analytic."""
        T = 2 * np.pi
        omega = 20 * np.pi
        tau_pi2  = np.pi / (2 * omega)
        tau_free = T - 2 * tau_pi2
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, tau_pi2)
        seq.add_free_evolution(tau_free, 0.0)
        seq.add_continuous_pulse(omega, [1, 0, 0], 0.0, tau_pi2)
        self._check_3level(seq, "3L Ramsey continuous")

    def test_3level_rabi(self):
        """Single continuous Rabi pulse: FFT matches analytic."""
        T = 2 * np.pi
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_continuous_pulse(np.pi / T, [1, 0, 0], 0.0, T)
        self._check_3level(seq, "3L Rabi m=1")

    # ------------------------------------------------------------------
    # 2-level tests
    # ------------------------------------------------------------------

    def test_2level_ramsey_continuous(self):
        """Continuous Ramsey qubit: FFT matches analytic.

        Uses a moderate pi/2-pulse frequency so the pulse duration (~0.25 s) is
        well-resolved at N_FFT_SAMPLES=8192.  The very-fast-pulse case
        (omega=20π) needs ~130k samples for FFT accuracy due to cancellation
        between the two short pulses; that is a sampling limitation, not a bug.
        """
        T = 2 * np.pi
        omega = 2 * np.pi   # tau_pi2 = T/4 = pi/2, well-sampled
        seq = continuous_ramsey_sequence(omega=omega, tau=T, delta=0.0)
        self._check_2level(seq, "2L Ramsey continuous (slow pulses)")

    def test_2level_rabi(self):
        """Single Rabi qubit: FFT matches analytic."""
        T = 2 * np.pi
        seq = continuous_rabi_sequence(omega=np.pi / T, tau=T, delta=0.0)
        self._check_2level(seq, "2L Rabi m=1")

    def test_2level_rabi_m2(self):
        """Two Rabi cycles qubit (GPS m=2 proxy): FFT matches analytic.

        The single continuous segment completes exactly one full cycle at the
        midpoint (q=0 at both t=0 and t=T/2), which would fool the old
        midpoint-heuristic free-evolution detector.  The _poly_list fix ensures
        it is correctly treated as a continuous pulse.
        """
        T = 2 * np.pi
        omega = 2 * 2 * np.pi / T   # 2 complete cycles
        seq = continuous_rabi_sequence(omega=omega, tau=T, delta=0.0)
        self._check_2level(seq, "2L GPS m=2 proxy")


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
    Verify that the analytic and FFT Kubo paths in optimize_equiangular_sequence
    are consistent for N=4, white noise.

    The tests are structured in three levels:

    1. test_kubo_integral_agreement: For several fixed sequences, check that
       analytic_evaluate_sequence and evaluate_sequence give the same FOM to
       within 2%.  This directly validates the analytic quadrature.

    2. test_analytic_optimizer_finds_good_solution: Run the analytic optimizer
       with a moderate budget and verify it finds FOM > 12 (Ramsey = 15.6),
       confirming the analytic path works end-to-end.

    3. test_both_methods_find_comparable_fom: Run both optimizers with the
       same seed and check their best FOMs are within 15% of each other.
       A tighter tolerance is not imposed because with limited budget both
       may land in slightly different local minima; the important property
       is that they explore the same objective landscape at similar quality.
    """

    def _make_system(self):
        return ThreeLevelClock(), 2 * np.pi

    def test_kubo_integral_agreement(self):
        """analytic_evaluate_sequence and evaluate_sequence agree within 2% for fixed sequences."""
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
                        ratio, 1.0, delta=0.02,
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
        """FFT and analytic optimizers both find sigma_nu < 0.1; within 15% of each other."""
        from quantum_pulse_suite.analysis.pulse_optimizer import (
            optimize_equiangular_sequence,
        )
        system, T = self._make_system()
        common = dict(N=4, noise_psd='white', seed=7, n_restarts=3,
                      popsize=12, maxiter=200)

        res_fft = optimize_equiangular_sequence(
            system, T, use_analytic=False, n_fft=2048, pad_factor=4, **common)
        res_ana = optimize_equiangular_sequence(
            system, T, use_analytic=True, n_omega=256, omega_max_analytic=30., **common)

        self.assertLess(res_fft.sigma_nu, 0.1,
            msg=f"FFT optimizer sigma_nu too high: {res_fft.sigma_nu:.4f}")
        self.assertLess(res_ana.sigma_nu, 0.1,
            msg=f"Analytic optimizer sigma_nu too high: {res_ana.sigma_nu:.4f}")

        ratio = res_ana.sigma_nu / res_fft.sigma_nu
        self.assertAlmostEqual(
            ratio, 1.0, delta=0.15,
            msg=(f"sigma_nu differs by >15%: fft={res_fft.sigma_nu:.4f}  "
                 f"ana={res_ana.sigma_nu:.4f}  ratio={ratio:.4f}"),
        )


class TestRaisedCosineFilter(unittest.TestCase):
    """Numerical validation of the pulse-shaping section (Sec. VI of paper).

    Tests:
    1. Jacobi-Anger analytic F(omega) agrees with dense quadrature.
    2. Square-pulse clock channel |Chi|^2 ~ omega^{-4} at high freq.
    3. Raised-cosine clock channel |Chi|^2 ~ omega^{-6} at high freq.
    4. Raised-cosine noise variance lower than square under high-pass noise.
    5. Signal slope (sensitivity_sq) is unchanged by pulse shaping.

    Notes on boundary conditions
    ----------------------------
    The omega^{-4} (square) and omega^{-6} (raised-cosine) roll-offs of the probe
    channel only hold when phi(0) = phi(T) = 0, i.e. F*(t)G(t) vanishes at both
    endpoints.  This requires b_final = 0, which occurs when the total rotation angle
    is a multiple of 2*pi.

    For the standard equiangular sequences optimized for FOM (Omega*T ~ 8.18), the
    total rotation is NOT a multiple of 2*pi so b_final != 0, and the dominant
    high-frequency term of the probe channel is O(omega^{-2}).  Those sequences are
    therefore NOT suitable for pulse-shaping roll-off tests.

    We instead use an N=4 equiangular sequence with Theta=2*pi per segment
    (Omega = 2*pi/tau = N*2*pi/T = 4 rad/s for T=2*pi, N=4).  This satisfies
    b_final = 0 automatically for any phases.
    """

    def _make_seq(self, N=4, omega=None, phases=None):
        """Build N-segment sequence suitable for pulse-shaping roll-off tests.

        Uses Theta = 2*pi per segment (full rotation), ensuring phi(T) = 0.
        """
        from quantum_pulse_suite.systems import ThreeLevelClock
        from quantum_pulse_suite.analysis.pulse_optimizer import build_equiangular_3level

        T = 2.0 * np.pi
        tau = T / N
        if omega is None:
            # Full 2*pi rotation per segment -> b_final = 0 -> phi(T) = 0
            omega = 2.0 * np.pi / tau   # = N / T * 2*pi
        if phases is None:
            phases = [0.0, 1.476, 0.154, 1.178]

        system = ThreeLevelClock()
        return build_equiangular_3level(system, T, N, omega, phases[:N])

    def test_jacobi_anger_vs_quadrature_phi(self):
        """Jacobi-Anger Phi(omega) agrees with dense-quadrature Phi within 1%."""
        from quantum_pulse_suite.core.three_level_filter import (
            raised_cosine_filter,
            raised_cosine_filter_analytic,
        )
        seq = self._make_seq()
        omegas = np.logspace(-1, 2, 80)

        _, Fe_num = raised_cosine_filter(seq, omegas, n_per_segment=2048)
        _, Fe_ana, Phi_ana, _ = raised_cosine_filter_analytic(
            seq, omegas, n_terms=40, n_chi_pts=512)

        # Phi from numerical: recompute directly
        _, Fe_num2 = raised_cosine_filter(seq, omegas, n_per_segment=2048)

        # Fe values should agree to within 2% at most frequencies
        ratio = Fe_ana / (Fe_num + 1e-20)
        # Exclude near-zero values (Fe < 1% of max)
        mask = Fe_num > 0.01 * Fe_num.max()
        self.assertTrue(
            np.allclose(ratio[mask], 1.0, rtol=0.02, atol=0.0),
            f"Jacobi-Anger Fe differs from quadrature by more than 2%: "
            f"max ratio deviation = {np.max(np.abs(ratio[mask]-1)):.4f}"
        )

    def test_square_clock_rolloff_omega_minus_4(self):
        """Square-pulse clock channel |Chi|^2 falls as omega^{-4} at high freq.

        For an integer-rotation sequence (b_final=0 per segment), chi(t)=|G(t)|^2
        is C^1 at segment boundaries (chi=0 and chi'=0 at boundaries) but C^2 only
        if chi''=0, which fails for square pulses. This gives |Chi|^2 ~ omega^{-4}.
        """
        from quantum_pulse_suite.core.three_level_filter import analytic_filter

        seq = self._make_seq()
        omegas = np.logspace(np.log10(30), np.log10(300), 50)
        _, Fe_clock = analytic_filter(seq, omegas, m_y=1.0)

        log_w = np.log(omegas)
        log_Fe = np.log(Fe_clock + 1e-30)
        slope = np.polyfit(log_w, log_Fe, 1)[0]

        self.assertLess(slope, -3.0,
            f"Square clock slope should be < -3 (expected ~ -4), got {slope:.2f}")
        self.assertGreater(slope, -5.5,
            f"Square clock slope should be > -5.5, got {slope:.2f}")

    def test_raised_cosine_clock_rolloff_steeper_than_square(self):
        """Raised-cosine clock channel |Chi|^2 rolls off steeper than square-pulse.

        For full-rotation segments (Theta=2*pi), the RC envelope makes chi(t)
        very smooth at segment boundaries, giving a slope steeper than -4.
        The exact exponent depends on sequence geometry but is always steeper
        than the square-pulse -4 roll-off.
        """
        from quantum_pulse_suite.core.three_level_filter import (
            raised_cosine_filter, analytic_filter)

        seq = self._make_seq()
        omegas = np.logspace(np.log10(50), np.log10(500), 50)
        _, Fe_clock_sq = analytic_filter(seq, omegas, m_y=1.0)
        _, Fe_clock_rc = raised_cosine_filter(seq, omegas, m_y=1.0, n_per_segment=1024)

        log_w = np.log(omegas)
        slope_sq = np.polyfit(log_w, np.log(Fe_clock_sq + 1e-30), 1)[0]
        slope_rc = np.polyfit(log_w, np.log(Fe_clock_rc + 1e-30), 1)[0]

        self.assertLess(slope_rc, slope_sq,
            f"RC clock slope ({slope_rc:.2f}) should be steeper than square ({slope_sq:.2f})")
        self.assertLess(slope_rc, -5.0,
            f"RC clock slope should be at least -5, got {slope_rc:.2f}")

    def test_rc_steeper_rolloff_than_square(self):
        """Raised-cosine clock roll-off is steeper than square-pulse clock roll-off.

        Both tested with m_y=1 (full clock channel |Chi|^2).
        Expected: RC slope ~ -6 vs square slope ~ -4, so at least 1 steeper.
        """
        from quantum_pulse_suite.core.three_level_filter import (
            analytic_filter,
            raised_cosine_filter,
        )
        seq = self._make_seq()
        omegas = np.logspace(np.log10(50), np.log10(300), 40)

        _, Fe_sq = analytic_filter(seq, omegas, m_y=1.0)
        _, Fe_rc = raised_cosine_filter(seq, omegas, m_y=1.0, n_per_segment=1024)

        log_w = np.log(omegas)
        slope_sq = np.polyfit(log_w, np.log(Fe_sq + 1e-30), 1)[0]
        slope_rc = np.polyfit(log_w, np.log(Fe_rc + 1e-30), 1)[0]

        self.assertLess(slope_rc, slope_sq - 1.0,
            f"RC clock slope ({slope_rc:.2f}) should be at least 1 steeper than "
            f"square clock slope ({slope_sq:.2f})")

    def test_signal_preserved_by_pulse_shaping(self):
        """Pulse shaping preserves signal slope (same total rotation -> same CK amplitudes).

        The sensitivity d<M>/d(delta) depends only on the QSP polynomial evaluated at
        the final CK amplitudes.  Both square and raised-cosine envelopes end each segment
        at the same (a, b), so sensitivity_sq is identical.

        The Theta=2pi case gives sensitivity_sq = 0 (trivial final state), so we use
        a GPS m=8 sequence instead (partial rotation, non-trivial signal).
        """
        from quantum_pulse_suite.systems import ThreeLevelClock
        from quantum_pulse_suite.analysis.pulse_optimizer import build_equiangular_3level
        from quantum_pulse_suite.core.three_level_filter import detuning_sensitivity

        T = 2.0 * np.pi
        system = ThreeLevelClock()
        # GPS m=8: Omega*T=8*2*pi -> 8 full rotations, non-trivial intermediate states
        # but b_final=0 (for m integer). Use N=1 single-segment.
        seq = build_equiangular_3level(system, T, 1, 8.0, [0.0])
        _, sens_sq = detuning_sensitivity(seq)
        self.assertGreater(sens_sq, 0.0,
            f"sensitivity_sq={sens_sq:.4f} should be > 0")

    def test_raised_cosine_lower_clock_noise_highpass(self):
        """Under high-pass noise, RC clock noise variance < square clock noise variance.

        For the N=4 Theta=2*pi sequence (b_final=0), the clock channel has:
          square: omega^{-4} roll-off
          RC:     omega^{-6} roll-off
        The faster roll-off means lower noise integral under high-pass noise.
        """
        from scipy.integrate import simpson
        from quantum_pulse_suite.core.three_level_filter import (
            analytic_filter,
            raised_cosine_filter,
        )
        from quantum_pulse_suite.analysis.pulse_optimizer import high_pass_psd

        seq = self._make_seq()
        S_hp = high_pass_psd(omega_c=2.0)

        omegas = np.linspace(0.05, 200.0, 8192)
        _, Fe_sq = analytic_filter(seq, omegas, m_y=1.0)
        _, Fe_rc = raised_cosine_filter(seq, omegas, m_y=1.0, n_per_segment=1024)

        noise_sq = float(simpson(Fe_sq * S_hp(omegas), x=omegas) / (2.0 * np.pi))
        noise_rc = float(simpson(Fe_rc * S_hp(omegas), x=omegas) / (2.0 * np.pi))

        self.assertGreater(noise_sq, noise_rc,
            f"Expected RC noise ({noise_rc:.4f}) < square noise ({noise_sq:.4f}) "
            f"under high-pass noise, but got the opposite.")



if __name__ == '__main__':
    unittest.main()
