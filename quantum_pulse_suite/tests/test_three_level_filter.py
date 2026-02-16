"""
Unit tests for three-level clock filter functions.

Tests cover:
- FFT vs analytic agreement for Phi(w) on Ramsey and spin echo sequences
- Known limits for Ramsey with delta=0
- Ff formula correctness (sinc-like, analytic)
- m_z dependence of Fe
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
)


class TestFfAnalytic(unittest.TestCase):
    """Test the protocol-independent f-noise filter function Ff."""

    def test_Ff_at_m_z_zero(self):
        """For m_z=0 (sigma_x measurement), Ff = -(1-cos wT)/w^2."""
        T = 2.0
        freqs = np.linspace(0.5, 50.0, 200)
        Ff = Ff_analytic(freqs, T, m_z=0.0)
        expected = -(1 - np.cos(freqs * T)) / freqs**2
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
        """At w->0, Ff -> (m_z^2 - 1)*T^2/2."""
        T = 3.0
        m_z = 0.5
        freqs = np.array([1e-14, 1e-13])
        Ff = Ff_analytic(freqs, T, m_z)
        expected = (m_z**2 - 1) * T**2 / 2
        np.testing.assert_allclose(Ff, expected, rtol=1e-6)

    def test_Ff_positive_freqs_sign(self):
        """For 0 < |m_z| < 1, Ff should be non-positive at positive freqs."""
        T = 2.0
        freqs = np.linspace(0.1, 50.0, 500)
        for m_z in [0.0, 0.3, 0.7]:
            Ff = Ff_analytic(freqs, T, m_z)
            # (m_z^2 - 1) < 0 and (1-cos)/w^2 >= 0, so Ff <= 0
            self.assertTrue(np.all(Ff <= 1e-15),
                            f"Ff should be <= 0 for m_z={m_z}")

    def test_Ff_scaling_with_T(self):
        """Ff peak scales as T^2."""
        m_z = 0.0
        for T in [1.0, 2.0, 4.0]:
            freqs = np.array([1e-11])
            Ff = Ff_analytic(freqs, T, m_z)
            expected = -(T**2) / 2
            np.testing.assert_allclose(Ff, expected, rtol=1e-6,
                                       err_msg=f"T={T} scaling wrong")


class TestFFTThreeLevelFilter(unittest.TestCase):
    """Test FFT-based three-level filter function."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_fft_returns_correct_shapes(self):
        """FFT returns arrays of consistent shapes."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, Ff, Phi = fft_three_level_filter(seq, n_samples=1024)

        self.assertEqual(len(Fe), len(freqs))
        self.assertEqual(len(Ff), len(freqs))
        self.assertEqual(len(Phi), len(freqs))
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

    def test_fft_Phi_finite(self):
        """Phi values should be finite."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        _, _, _, Phi = fft_three_level_filter(seq)
        self.assertTrue(np.all(np.isfinite(Phi)))

    def test_fft_Fe_varies_with_m_z(self):
        """Fe should change with m_z (different prefactor)."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        _, Fe0, _, _ = fft_three_level_filter(seq, m_z=0.0)
        _, Fe1, _, _ = fft_three_level_filter(seq, m_z=1.0)

        # For m_z=0: prefactor = 1/2*(1+0) = 0.5
        # For m_z=1: prefactor = 1/2*(0+2) = 1.0
        # So Fe1 = 2 * Fe0
        mask = Fe0 > 1e-10 * np.max(Fe0)
        if np.any(mask):
            ratio = Fe1[mask] / Fe0[mask]
            np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)


class TestFFTvsAnalytic(unittest.TestCase):
    """Compare FFT and analytic three-level filter functions."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def _compare_Phi(self, seq, label, rtol=0.01):
        """Helper: compare |Phi|^2 between FFT and analytic methods."""
        seq.compute_polynomials()

        freqs_fft, Fe_fft, _, Phi_fft = fft_three_level_filter(
            seq, n_samples=8192, pad_factor=4, m_z=0.0)

        _, Fe_ana, _, Phi_ana = analytic_three_level_filter(
            seq, freqs_fft, m_z=0.0)

        # Compare |Phi|^2
        Phi2_fft = np.abs(Phi_fft)**2
        Phi2_ana = np.abs(Phi_ana)**2

        T = seq.total_duration()
        mask = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0)
        peak = max(np.max(Phi2_fft[mask]), np.max(Phi2_ana[mask]))

        if peak < 1e-20:
            return  # nothing to compare

        sig = (Phi2_fft[mask] > 1e-4 * peak) & (Phi2_ana[mask] > 1e-4 * peak)
        if not np.any(sig):
            return

        rel_err = np.abs(Phi2_fft[mask][sig] - Phi2_ana[mask][sig]) / Phi2_ana[mask][sig]
        self.assertLess(
            np.max(rel_err), rtol,
            msg=f"{label}: max |Phi|^2 rel error {np.max(rel_err):.4f} > {rtol}"
        )

    def test_ramsey_fft_vs_analytic(self):
        """Ramsey: FFT |Phi|^2 matches analytic."""
        for tau in [1.0, 2.0]:
            seq = multilevel_ramsey(self.system, self.system.probe,
                                    tau=tau, delta=0.0)
            self._compare_Phi(seq, f"Ramsey(tau={tau})")

    def test_ramsey_with_detuning_fft_vs_analytic(self):
        """Ramsey with detuning: FFT |Phi|^2 matches analytic."""
        seq = multilevel_ramsey(self.system, self.system.probe,
                                tau=2.0, delta=0.5)
        self._compare_Phi(seq, "Ramsey(tau=2, delta=0.5)")

    def test_spin_echo_fft_vs_analytic(self):
        """Spin echo: FFT |Phi|^2 matches analytic."""
        for tau in [1.0, 2.0]:
            seq = multilevel_spin_echo(self.system, self.system.probe,
                                        tau=tau, delta=0.0)
            self._compare_Phi(seq, f"SpinEcho(tau={tau})")

    def test_cpmg_fft_vs_analytic(self):
        """CPMG: FFT |Phi|^2 matches analytic."""
        tau = 2.0
        for n in [1, 2, 4]:
            seq = multilevel_cpmg(self.system, self.system.probe,
                                   tau=tau, n_pulses=n, delta=0.0)
            self._compare_Phi(seq, f"CPMG-{n}(tau={tau})")


class TestRamseyKnownLimits(unittest.TestCase):
    """Test Phi against known formulas for simple sequences."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_ramsey_delta_zero_Phi_form(self):
        """
        Ramsey delta=0: the QSP unitary on {|g>,|e>} is
        pi/2_x - free(tau) - pi/2_x.

        The Phi integrand F*(t)G(t) should be nonzero during free evolution
        (since f,g != 0 after first pi/2), and the integral gives a sinc-like
        shape.
        """
        tau = 2.0
        seq = multilevel_ramsey(self.system, self.system.probe,
                                tau=tau, delta=0.0)
        seq.compute_polynomials()

        # After first pi/2_x: f = cos(pi/4) = 1/sqrt(2), g = sin(pi/4) = 1/sqrt(2)
        # During free evolution (delta=0): F(t) = f, G(t) = g (constant)
        # So F*G = conj(f)*g = (1/sqrt(2))*(1/sqrt(2)) = 1/2

        # Phi(w) = (1/2) * integral_0^tau e^{-iwt} dt
        # = (1/2) * (1 - e^{-iwτ})/(iw)
        # |Phi|^2 = (1/4) * 2(1-cos(wτ))/w^2 = (1-cos(wτ))/(2w^2)

        freqs = np.linspace(0.5, 30.0, 200)
        _, _, _, Phi_ana = analytic_three_level_filter(seq, freqs, m_z=0.0)

        Phi2_expected = (1 - np.cos(freqs * tau)) / (2 * freqs**2)
        Phi2_actual = np.abs(Phi_ana)**2

        np.testing.assert_allclose(Phi2_actual, Phi2_expected, rtol=0.01,
                                    err_msg="Ramsey delta=0 |Phi|^2 mismatch")

    def test_ramsey_delta_nonzero_Phi(self):
        """
        Ramsey with detuning: F*G is still constant during free evolution
        (detuning phases cancel in the product), so |Phi|^2 has the same
        sinc envelope regardless of delta.
        """
        tau = 2.0
        freqs = np.linspace(0.5, 30.0, 200)

        seq0 = multilevel_ramsey(self.system, self.system.probe,
                                  tau=tau, delta=0.0)
        seq0.compute_polynomials()
        _, _, _, Phi0 = analytic_three_level_filter(seq0, freqs, m_z=0.0)

        seq1 = multilevel_ramsey(self.system, self.system.probe,
                                  tau=tau, delta=1.5)
        seq1.compute_polynomials()
        _, _, _, Phi1 = analytic_three_level_filter(seq1, freqs, m_z=0.0)

        # |Phi|^2 should be the same regardless of delta
        np.testing.assert_allclose(
            np.abs(Phi0)**2, np.abs(Phi1)**2, rtol=0.01,
            err_msg="|Phi|^2 should be delta-independent for Ramsey")


class TestMzDependence(unittest.TestCase):
    """Test m_z dependence of Fe and Ff."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_Fe_prefactor_values(self):
        """Fe prefactor = (1 - m_z^2 + 2*m_z)/2 for different m_z values."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)

        # Get Phi (same for all m_z)
        freqs, _, _, Phi_ref = fft_three_level_filter(seq, m_z=0.0)
        Phi2 = np.abs(Phi_ref)**2

        for m_z in [0.0, 0.5, 1.0, -0.5]:
            _, Fe, _, _ = fft_three_level_filter(seq, m_z=m_z)
            expected_prefactor = 0.5 * ((1 - m_z**2) + 2 * m_z)
            Fe_expected = expected_prefactor * Phi2

            mask = Fe_expected > 1e-10 * np.max(Fe_expected)
            if np.any(mask):
                np.testing.assert_allclose(
                    Fe[mask], Fe_expected[mask], rtol=1e-6,
                    err_msg=f"Fe prefactor wrong for m_z={m_z}")

    def test_Ff_varies_with_m_z(self):
        """Ff should scale as (m_z^2 - 1)."""
        T = 2.0
        freqs = np.linspace(1.0, 20.0, 100)

        Ff_0 = Ff_analytic(freqs, T, m_z=0.0)  # coefficient = -1
        Ff_half = Ff_analytic(freqs, T, m_z=0.5)  # coefficient = -0.75

        np.testing.assert_allclose(Ff_half, 0.75 * Ff_0, rtol=1e-12)


class TestNoiseVarianceIntegration(unittest.TestCase):
    """Test noise variance computation."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_white_noise_e_only(self):
        """Variance with white e-noise only should be positive."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        freqs, Fe, Ff, _ = fft_three_level_filter(seq, m_z=0.0)

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
        # Ff <= 0 for m_z=0, so Sf*Ff contribution is non-positive
        # This means the f-noise actually decreases variance (measurement noise)
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
        freqs, Fe, Ff, _ = fft_three_level_filter(seq, m_z=1.0)

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


class TestSpinEchoPhiShape(unittest.TestCase):
    """Test that spin echo Phi has expected structure."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_spin_echo_zeros_at_harmonics(self):
        """
        Spin echo |Phi|^2 should have zeros at w = 2*pi*n/tau (even harmonics)
        like the qubit case.
        """
        tau = 2.0
        seq = multilevel_spin_echo(self.system, self.system.probe,
                                    tau=tau, delta=0.0)
        seq.compute_polynomials()

        # Even harmonics of the fundamental frequency
        fundamental = 2 * np.pi / tau
        # At w = 2*fundamental, 4*fundamental, etc. the filter should be small
        even_harmonics = np.array([2, 4, 6]) * fundamental
        _, _, _, Phi = analytic_three_level_filter(seq, even_harmonics, m_z=0.0)

        # These should be near zero (exact zeros for the switching function)
        np.testing.assert_allclose(
            np.abs(Phi)**2, 0.0, atol=1e-6,
            err_msg="Spin echo Phi should vanish at even harmonics")


class TestCPMGPhiPeakShift(unittest.TestCase):
    """Test that CPMG peak shifts with number of pulses."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_cpmg_peak_shifts_with_n(self):
        """More pi pulses should shift the main peak to higher frequency."""
        tau = 2.0
        freqs = np.linspace(0.5, 100.0, 2000)

        peaks = []
        for n in [1, 2, 4]:
            seq = multilevel_cpmg(self.system, self.system.probe,
                                   tau=tau, n_pulses=n, delta=0.0)
            seq.compute_polynomials()
            _, Fe, _, _ = analytic_three_level_filter(seq, freqs, m_z=0.0)
            peak_idx = np.argmax(Fe)
            peaks.append(freqs[peak_idx])

        # Peak frequency should increase with n
        self.assertGreater(peaks[1], peaks[0])
        self.assertGreater(peaks[2], peaks[1])


if __name__ == '__main__':
    unittest.main()
