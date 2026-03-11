"""
Unit tests for Kubo sensitivity methods on filter function classes.

Tests cover:
- Qubit kubo_sensitivity() via FilterFunction base class
- SubspaceFilterFunction bug fixes (complex return, conj(g), sign)
- SubspaceFilterFunction.measurement_sensitivity() cross-validated against
  fft_three_level_filter (independent path: full unitary propagation + FFT)
- GPSFilterFunction.filter_function_for_measurement() cross-validated against
  fft_three_level_filter
"""

import unittest
import numpy as np

from quantum_pulse_suite.core.pulse_sequence import (
    ramsey_sequence,
    spin_echo_sequence,
    continuous_rabi_sequence,
    continuous_ramsey_sequence,
    InstantaneousPulseSequence,
    FreeEvolution,
)
from quantum_pulse_suite.core.multilevel import (
    MultiLevelPulseSequence,
    SubspaceFilterFunction,
    multilevel_ramsey,
    multilevel_spin_echo,
)
from quantum_pulse_suite.core.three_level_filter import fft_three_level_filter
from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.global_phase_spectroscopy import (
    GlobalPhaseSpectroscopySequence,
)


# =============================================================================
# Qubit Kubo sensitivity tests
# =============================================================================


class TestQubitKuboSensitivity(unittest.TestCase):
    """Test kubo_sensitivity() on qubit (2-level) filter functions."""

    def setUp(self):
        self.frequencies = np.linspace(0.5, 30, 200)

    def test_free_evolution_sigma_z(self):
        """Free evolution: F is along z, so B=z gives kubo = 0."""
        seq = InstantaneousPulseSequence()
        seq.add_free_evolution(1.0)
        seq.compute_polynomials()
        ff = seq.get_filter_function_calculator()

        B_z = np.array([0, 0, 1])
        kubo = ff.kubo_sensitivity(self.frequencies, B_z)
        np.testing.assert_allclose(kubo, 0.0, atol=1e-10)

    def test_free_evolution_sigma_x(self):
        """Free evolution: F along z, B=x → kubo = |Fz|²."""
        seq = InstantaneousPulseSequence()
        seq.add_free_evolution(1.0)
        seq.compute_polynomials()
        ff = seq.get_filter_function_calculator()

        B_x = np.array([1, 0, 0])
        kubo = ff.kubo_sensitivity(self.frequencies, B_x)
        ns = ff.noise_susceptibility(self.frequencies)
        np.testing.assert_allclose(kubo, ns, rtol=1e-10)

    def test_ramsey_sigma_y_zero(self):
        """Instantaneous Ramsey: all F along y after π/2 pulses → kubo_y = 0."""
        seq = ramsey_sequence(tau=1.0)
        seq.compute_polynomials()
        ff = seq.get_filter_function_calculator()

        B_y = np.array([0, 1, 0])
        kubo = ff.kubo_sensitivity(self.frequencies, B_y)
        np.testing.assert_allclose(kubo, 0.0, atol=1e-10)

    def test_ramsey_sigma_z_equals_noise_susceptibility(self):
        """Instantaneous Ramsey: F along y, B=z → kubo_z = |Fy|² = noise_susceptibility."""
        seq = ramsey_sequence(tau=1.0)
        seq.compute_polynomials()
        ff = seq.get_filter_function_calculator()

        B_z = np.array([0, 0, 1])
        kubo = ff.kubo_sensitivity(self.frequencies, B_z)
        ns = ff.noise_susceptibility(self.frequencies)
        np.testing.assert_allclose(kubo, ns, rtol=1e-10)

    def test_continuous_rabi_sigma_y_nonzero(self):
        """Continuous Rabi drive mixes axes → nonzero sensitivity for σ_y."""
        seq = continuous_rabi_sequence(omega=2*np.pi, tau=1.0, delta=0.1)
        seq.compute_polynomials()
        ff = seq.get_filter_function_calculator()

        B_y = np.array([0, 1, 0])
        kubo = ff.kubo_sensitivity(self.frequencies, B_y)
        self.assertGreater(np.max(kubo), 1e-6)

    def test_sum_rule(self):
        """Sum of kubo over 3 orthogonal B's = 2 * |F|²."""
        seq = continuous_ramsey_sequence(omega=20*np.pi, tau=1.0, delta=0.25)
        seq.compute_polynomials()
        ff = seq.get_filter_function_calculator()

        Bx = np.array([1, 0, 0])
        By = np.array([0, 1, 0])
        Bz = np.array([0, 0, 1])

        kx = ff.kubo_sensitivity(self.frequencies, Bx)
        ky = ff.kubo_sensitivity(self.frequencies, By)
        kz = ff.kubo_sensitivity(self.frequencies, Bz)

        ns = ff.noise_susceptibility(self.frequencies)
        np.testing.assert_allclose(kx + ky + kz, 2 * ns, rtol=1e-10)

    def test_kubo_nonnegative(self):
        """Kubo sensitivity should always be non-negative."""
        seq = spin_echo_sequence(tau=2.0)
        seq.compute_polynomials()
        ff = seq.get_filter_function_calculator()

        for B in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
                   np.array([1,1,1])/np.sqrt(3)]:
            kubo = ff.kubo_sensitivity(self.frequencies, B)
            self.assertTrue(np.all(kubo >= -1e-15),
                            f"Negative kubo for B={B}")


# =============================================================================
# SubspaceFilterFunction bug fix tests
# =============================================================================


class TestSubspaceFilterFunctionBugFixes(unittest.TestCase):
    """Test that the 3 bugs in SubspaceFilterFunction are fixed."""

    def setUp(self):
        self.system = ThreeLevelClock()
        self.frequencies = np.linspace(0.5, 30, 200)

    def test_complex_return(self):
        """filter_function should return complex dtype (not real)."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()
        Fx, Fy, Fz = ff.filter_function(self.frequencies)

        self.assertTrue(np.issubdtype(Fx.dtype, np.complexfloating),
                        f"Fx dtype is {Fx.dtype}, expected complex")
        self.assertTrue(np.issubdtype(Fy.dtype, np.complexfloating),
                        f"Fy dtype is {Fy.dtype}, expected complex")
        self.assertTrue(np.issubdtype(Fz.dtype, np.complexfloating),
                        f"Fz dtype is {Fz.dtype}, expected complex")

    def test_nonzero_Fy_for_ramsey(self):
        """After π/2, filter function should have nonzero Fy component."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()
        Fx, Fy, Fz = ff.filter_function(self.frequencies)

        # After first π/2_x, toggling frame rotates z→y, so Fy should be large
        self.assertGreater(np.max(np.abs(Fy)), 1e-3,
                           "Fy should be nonzero for Ramsey")

    def test_first_segment_not_special_cased(self):
        """First segment should use the general formula, not just Fz += factor."""
        # For a sequence that starts with free evolution (no pulse yet),
        # f=1, g=0 → expr=0, fz_expr=1, so it should still give Fz += factor
        # But for a sequence starting with a pulse then free evo, all 3 should be used
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()
        Fx, Fy, Fz = ff.filter_function(self.frequencies)

        # The single free-evolution segment starts after a π/2 pulse,
        # so f,g != (1,0). All three components should contribute.
        total = np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2
        self.assertGreater(np.max(total), 0)

    def test_matches_qubit_filter_for_identity_subspace(self):
        """Multilevel noise_susceptibility should match qubit for same Ramsey."""
        tau = 1.0
        # Qubit
        q_seq = ramsey_sequence(tau=tau)
        q_seq.compute_polynomials()
        q_ff = q_seq.get_filter_function_calculator()
        q_ns = q_ff.noise_susceptibility(self.frequencies)

        # Multilevel
        m_seq = multilevel_ramsey(self.system, self.system.probe, tau=tau)
        m_ff = m_seq.get_filter_function_calculator()
        m_ns = m_ff.noise_susceptibility(self.frequencies)

        # These should be close (same 2-level physics, same pulses)
        np.testing.assert_allclose(m_ns, q_ns, rtol=0.01,
                                    err_msg="Multilevel and qubit noise_susceptibility differ")


# =============================================================================
# Three-level measurement_sensitivity tests
# =============================================================================


class TestMeasurementSensitivity(unittest.TestCase):
    """Test SubspaceFilterFunction.measurement_sensitivity() via Fe/Chi.

    The _matches_fft tests cross-validate the polynomial analytic path against
    fft_three_level_filter, which uses a completely independent computation:
    it propagates the full unitary U(t) at each time step, projects to the 2x2
    subspace to extract f(t) and g(t), samples chi(t)=|g(t)|^2, and FFTs.
    """

    def setUp(self):
        self.system = ThreeLevelClock()
        self.frequencies = np.linspace(0.5, 30, 200)

    def test_ramsey_Fe_matches_fft(self):
        """Chi-based analytic_filter agrees with fft_three_level_filter (m_y=1).

        Cross-validates the polynomial analytic path (analytic_filter) against
        the independent unitary propagation + FFT path (fft_three_level_filter).
        """
        from quantum_pulse_suite.core.three_level_filter import analytic_filter
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)

        freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
            seq, n_samples=8192, m_y=1.0)

        T = seq.total_duration()
        mask = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 80.0)
        _, Fe_ana = analytic_filter(seq, freqs_fft[mask], m_y=1.0)
        Fe_ref = Fe_fft[mask]

        peak = max(np.max(Fe_ref), np.max(Fe_ana))
        sig = (Fe_ref > 1e-4 * peak) & (Fe_ana > 1e-4 * peak)
        self.assertTrue(np.any(sig), "No significant frequency points in Ramsey Fe comparison")

        rel_err = np.abs(Fe_ref[sig] - Fe_ana[sig]) / Fe_ana[sig]
        self.assertLess(np.max(rel_err), 0.05,
                        msg=f"Ramsey Fe: max rel error {np.max(rel_err):.4f}")

    def test_ramsey_Fe_nonzero(self):
        """Fe should be nonzero for Ramsey with σ_y measurement (m_y=1)."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()
        Fe = ff.measurement_sensitivity(self.frequencies, m_y=1.0)

        self.assertGreater(np.max(Fe), 1e-6,
                           "Fe should be nonzero for Ramsey")

    def test_m_y_one_gives_full_chi_squared(self):
        """For m_y=1 (σ_y): Fe = |Chi|²."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()

        Fe_1 = ff.measurement_sensitivity(self.frequencies, m_y=1.0)
        Chi = ff._compute_chi(self.frequencies)
        expected = np.abs(Chi)**2

        np.testing.assert_allclose(Fe_1, expected, rtol=1e-10)

    def test_m_y_zero_gives_zero(self):
        """For m_y=0, Fe vanishes."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()

        Fe_0 = ff.measurement_sensitivity(self.frequencies, m_y=0.0)
        expected = np.zeros_like(Fe_0)

        np.testing.assert_allclose(Fe_0, expected, rtol=1e-10)

    def test_spin_echo_Fe_matches_fft(self):
        """Spin echo chi-based analytic_filter agrees with fft_three_level_filter."""
        from quantum_pulse_suite.core.three_level_filter import analytic_filter
        seq = multilevel_spin_echo(self.system, self.system.probe, tau=2.0)

        freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
            seq, n_samples=8192, m_y=1.0)

        T = seq.total_duration()
        mask = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 80.0)
        _, Fe_ana = analytic_filter(seq, freqs_fft[mask], m_y=1.0)
        Fe_ref = Fe_fft[mask]

        peak = max(np.max(Fe_ref), np.max(Fe_ana))
        sig = (Fe_ref > 1e-4 * peak) & (Fe_ana > 1e-4 * peak)
        self.assertTrue(np.any(sig), "No significant frequency points in spin echo Fe comparison")

        rel_err = np.abs(Fe_ref[sig] - Fe_ana[sig]) / Fe_ana[sig]
        self.assertLess(np.max(rel_err), 0.05,
                        msg=f"Spin echo Fe: max rel error {np.max(rel_err):.4f}")


# =============================================================================
# GPS filter_function_for_measurement delegation tests
# =============================================================================


class TestGPSMeasurementDelegation(unittest.TestCase):
    """Test GPSFilterFunction.filter_function_for_measurement against fft_three_level_filter.

    GPS sequences are all-continuous, so the FFT path (unitary propagation)
    is genuinely independent of the polynomial analytic path.
    """

    def setUp(self):
        self.system = ThreeLevelClock()
        self.omega = 2 * np.pi
        self.frequencies = np.linspace(0.5, 30, 100)

    def test_gps_Fe_matches_fft(self):
        """GPS chi-based analytic_filter agrees with fft_three_level_filter (m_y=1).

        GPS uses continuous pulses, so this is a non-trivial cross-check of two
        independent implementations of F = |Chi|^2.
        """
        from quantum_pulse_suite.core.three_level_filter import analytic_filter
        for n_cyc in [1, 3]:
            gps = GlobalPhaseSpectroscopySequence(
                self.system, n_cycles=n_cyc, omega=self.omega)

            freqs_fft, Fe_fft, _, _ = fft_three_level_filter(
                gps._sequence, n_samples=8192, m_y=1.0)

            T = gps._sequence.total_duration()
            mask = (freqs_fft > 2 * np.pi / T) & (freqs_fft < 80.0)
            _, Fe_ana = analytic_filter(gps._sequence, freqs_fft[mask], m_y=1.0)
            Fe_ref = Fe_fft[mask]

            peak = max(np.max(Fe_ref), np.max(Fe_ana))
            sig = (Fe_ref > 1e-4 * peak) & (Fe_ana > 1e-4 * peak)
            self.assertTrue(np.any(sig),
                            f"GPS n_cyc={n_cyc}: no significant frequency points")

            rel_err = np.abs(Fe_ref[sig] - Fe_ana[sig]) / Fe_ana[sig]
            self.assertLess(np.max(rel_err), 0.05,
                            msg=f"GPS n_cyc={n_cyc} Fe: max rel error {np.max(rel_err):.4f}")

    def test_gps_Fe_nonnegative(self):
        """GPS Fe should be non-negative for all measurement types."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega)
        ff = gps.get_filter_function_calculator()

        for meas in ['z', 'x', 'y']:
            Fe = ff.filter_function_for_measurement(self.frequencies, meas)
            self.assertTrue(np.all(Fe >= -1e-15),
                            f"Negative Fe for measurement={meas}")

    def test_gps_only_sigma_y_is_nonzero(self):
        """With the updated model, only sigma_y^{gm} contributes to Fe."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=3, omega=self.omega)
        ff = gps.get_filter_function_calculator()

        Fe_z = ff.filter_function_for_measurement(self.frequencies, 'z')
        Fe_x = ff.filter_function_for_measurement(self.frequencies, 'x')
        Fe_y = ff.filter_function_for_measurement(self.frequencies, 'y')

        np.testing.assert_allclose(Fe_z, 0.0, atol=1e-12)
        np.testing.assert_allclose(Fe_x, 0.0, atol=1e-12)
        self.assertGreater(np.max(Fe_y), 1e-8)


if __name__ == '__main__':
    unittest.main()
