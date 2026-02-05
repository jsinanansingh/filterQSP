"""
Unit tests for Global Phase Spectroscopy module.

Tests the GPS protocol where continuous Rabi oscillation on the optical
transition is followed by measurement on the metastable (clock) transition.
"""

import unittest
import numpy as np

from quantum_pulse_suite.systems import ThreeLevelClock
from quantum_pulse_suite.analysis.global_phase_spectroscopy import (
    GlobalPhaseSpectroscopySequence,
    GPSFilterFunction,
    gps_filter_functions_comparison,
)


class TestGPSSequenceBasics(unittest.TestCase):
    """Test basic GPS sequence creation and properties."""

    def setUp(self):
        self.system = ThreeLevelClock()
        self.omega = 2 * np.pi  # 1 Hz Rabi frequency
        self.n_cycles = 5

    def test_sequence_creation(self):
        """Test GPS sequence can be created."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=self.n_cycles, omega=self.omega
        )

        self.assertEqual(gps.n_cycles, self.n_cycles)
        self.assertEqual(gps.omega, self.omega)
        self.assertEqual(gps.delta, 0.0)

    def test_total_time(self):
        """Test total time is m complete Rabi cycles."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=self.n_cycles, omega=self.omega
        )

        # T_total = m * 2π/Ω
        expected_time = self.n_cycles * 2 * np.pi / self.omega
        self.assertAlmostEqual(gps.total_time, expected_time, places=10)

    def test_cycle_period(self):
        """Test cycle period calculation."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=self.n_cycles, omega=self.omega
        )

        expected_period = 2 * np.pi / self.omega
        self.assertAlmostEqual(gps.cycle_period, expected_period, places=10)

    def test_with_detuning(self):
        """Test GPS sequence with non-zero detuning."""
        delta = 0.1
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=self.n_cycles, omega=self.omega, delta=delta
        )

        self.assertEqual(gps.delta, delta)


class TestGPSStateEvolution(unittest.TestCase):
    """Test state evolution through GPS sequence."""

    def setUp(self):
        self.system = ThreeLevelClock()
        self.omega = 2 * np.pi

    def test_evolve_clock_superposition(self):
        """Test evolving clock superposition through GPS."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=1, omega=self.omega
        )

        psi0 = self.system.prepare_clock_superposition()
        psi_final = gps.evolve_state(psi0)

        # Final state should be normalized
        norm = np.linalg.norm(psi_final)
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_complete_cycles_return(self):
        """Test that complete Rabi cycles return |g> to |g> (up to phase)."""
        # For integer cycles on probe, |g> should return to |g>
        # (The |m> component is unchanged)
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=1, omega=self.omega
        )

        # Start in ground state
        psi0 = self.system.prepare_state('ground')
        psi_final = gps.evolve_state(psi0)

        # After 1 complete cycle, should return to |g> (up to phase)
        pop_g = np.abs(psi_final[0])**2
        self.assertAlmostEqual(pop_g, 1.0, places=5)

    def test_metastable_unchanged(self):
        """Test that |m> component is unchanged by probe driving."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega
        )

        # Start in metastable state
        psi0 = self.system.prepare_state('metastable')
        psi_final = gps.evolve_state(psi0)

        # Should still be in |m>
        pop_m = np.abs(psi_final[2])**2
        self.assertAlmostEqual(pop_m, 1.0, places=10)

    def test_differential_phase_finite(self):
        """Test differential phase is computable and finite."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=3, omega=self.omega
        )

        phase = gps.differential_phase()
        self.assertTrue(np.isfinite(phase))

    def test_contrast_after_gps(self):
        """Test contrast is preserved for complete cycles without detuning."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=1, omega=self.omega, delta=0.0
        )

        initial_contrast = 1.0  # Perfect superposition
        final_contrast = gps.contrast()

        # For complete cycles without detuning, contrast should be preserved
        self.assertAlmostEqual(final_contrast, initial_contrast, places=5)


class TestGPSFilterFunction(unittest.TestCase):
    """Test GPS filter function computation."""

    def setUp(self):
        self.system = ThreeLevelClock()
        self.omega = 2 * np.pi
        self.frequencies = np.linspace(0.1, 50, 100)

    def test_filter_function_computable(self):
        """Test filter function can be computed."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega
        )
        ff = gps.get_filter_function_calculator()

        Fx, Fy, Fz = ff.filter_function(self.frequencies)

        self.assertEqual(len(Fx), len(self.frequencies))
        self.assertEqual(len(Fy), len(self.frequencies))
        self.assertEqual(len(Fz), len(self.frequencies))

    def test_filter_function_finite(self):
        """Test filter function values are finite."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega
        )
        ff = gps.get_filter_function_calculator()

        Fx, Fy, Fz = ff.filter_function(self.frequencies)

        self.assertTrue(np.all(np.isfinite(Fx)))
        self.assertTrue(np.all(np.isfinite(Fy)))
        self.assertTrue(np.all(np.isfinite(Fz)))

    def test_filter_function_nonnegative(self):
        """Test filter function squared is non-negative."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega
        )
        ff = gps.get_filter_function_calculator()

        for meas_type in ['z', 'x', 'y']:
            ff_sq = ff.filter_function_for_measurement(self.frequencies, meas_type)
            self.assertTrue(np.all(ff_sq >= -1e-10),
                           f"Negative values for measurement {meas_type}")

    def test_filter_function_z_measurement(self):
        """Test filter function for sigma_z measurement."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega
        )
        ff = gps.get_filter_function_calculator()

        ff_z = ff.filter_function_for_measurement(self.frequencies, 'z')

        # Should have some structure (not flat)
        self.assertGreater(np.std(ff_z), 0)

    def test_filter_function_x_measurement(self):
        """Test filter function for sigma_x measurement."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega
        )
        ff = gps.get_filter_function_calculator()

        ff_x = ff.filter_function_for_measurement(self.frequencies, 'x')

        self.assertTrue(np.all(np.isfinite(ff_x)))

    def test_characteristic_frequencies(self):
        """Test characteristic frequency identification."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=self.omega
        )
        ff = gps.get_filter_function_calculator()

        char_freqs = ff.characteristic_frequencies()

        # Should include Rabi frequency and harmonics
        self.assertIn(self.omega, char_freqs)
        self.assertTrue(len(char_freqs) > 1)

    def test_more_cycles_narrower_peaks(self):
        """Test that more cycles give narrower filter function peaks."""
        gps_few = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=2, omega=self.omega
        )
        gps_many = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=10, omega=self.omega
        )

        ff_few = gps_few.get_filter_function_calculator()
        ff_many = gps_many.get_filter_function_calculator()

        # Fine frequency grid around Rabi frequency
        freqs_fine = np.linspace(self.omega * 0.5, self.omega * 1.5, 200)

        ff_few_z = ff_few.filter_function_for_measurement(freqs_fine, 'z')
        ff_many_z = ff_many.filter_function_for_measurement(freqs_fine, 'z')

        # Normalize by peak
        if np.max(ff_few_z) > 0 and np.max(ff_many_z) > 0:
            ff_few_norm = ff_few_z / np.max(ff_few_z)
            ff_many_norm = ff_many_z / np.max(ff_many_z)

            # More cycles should have narrower peak (smaller FWHM)
            # Check that the normalized distribution is more concentrated
            # by comparing values at the edges
            edge_idx = [0, -1]
            avg_edge_few = np.mean([ff_few_norm[i] for i in edge_idx])
            avg_edge_many = np.mean([ff_many_norm[i] for i in edge_idx])

            # Many cycles should have smaller edge values (narrower peak)
            # Allow some tolerance due to numerical effects
            self.assertLessEqual(avg_edge_many, avg_edge_few + 0.1)


class TestGPSMeasurementTypes(unittest.TestCase):
    """Test different measurement operators."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_measurement_operators_hermitian(self):
        """Test all measurement operators are Hermitian."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=1, omega=2*np.pi
        )

        for meas_type in ['z', 'x', 'y', 'population_g', 'population_m']:
            op = gps.measurement_sensitivity(meas_type)
            np.testing.assert_array_almost_equal(
                op, op.conj().T,
                err_msg=f"Measurement operator {meas_type} not Hermitian"
            )

    def test_population_operators_are_projectors(self):
        """Test population operators are projectors (P^2 = P)."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=1, omega=2*np.pi
        )

        for meas_type in ['population_g', 'population_m']:
            P = gps.measurement_sensitivity(meas_type)
            P_squared = P @ P
            np.testing.assert_array_almost_equal(
                P, P_squared,
                err_msg=f"{meas_type} is not a projector"
            )

    def test_sigma_z_eigenvalues(self):
        """Test sigma_z measurement has correct eigenvalues."""
        gps = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=1, omega=2*np.pi
        )

        sz = gps.measurement_sensitivity('z')
        eigenvalues = np.sort(np.linalg.eigvalsh(sz))

        # Clock sigma_z: |g><g| - |m><m|
        # Eigenvalues: +1 (|g>), 0 (|e>), -1 (|m>)
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_almost_equal(eigenvalues, expected)


class TestGPSComparison(unittest.TestCase):
    """Test GPS filter function comparison utilities."""

    def setUp(self):
        self.system = ThreeLevelClock()
        self.omega = 2 * np.pi
        self.frequencies = np.linspace(0.1, 30, 50)

    def test_comparison_function(self):
        """Test filter function comparison utility."""
        n_cycles_list = [1, 3, 5]

        results = gps_filter_functions_comparison(
            self.system,
            n_cycles_list=n_cycles_list,
            omega=self.omega,
            frequencies=self.frequencies
        )

        self.assertIn('frequencies', results)
        self.assertIn('n_cycles', results)
        self.assertIn('filter_functions', results)

        for n in n_cycles_list:
            self.assertIn(n, results['filter_functions'])

    def test_comparison_includes_measurement_types(self):
        """Test comparison includes different measurement types."""
        results = gps_filter_functions_comparison(
            self.system,
            n_cycles_list=[3],
            omega=self.omega,
            frequencies=self.frequencies,
            measurement_types=['z', 'x']
        )

        self.assertIn('z', results['filter_functions'][3])
        self.assertIn('x', results['filter_functions'][3])

    def test_total_times_recorded(self):
        """Test total times are recorded in comparison."""
        n_cycles_list = [2, 4]

        results = gps_filter_functions_comparison(
            self.system,
            n_cycles_list=n_cycles_list,
            omega=self.omega,
            frequencies=self.frequencies
        )

        for n in n_cycles_list:
            expected_time = n * 2 * np.pi / self.omega
            self.assertAlmostEqual(
                results['total_times'][n], expected_time, places=10
            )


class TestGPSPhysics(unittest.TestCase):
    """Test physical predictions of GPS protocol."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_detuning_sensitivity(self):
        """Test that GPS is sensitive to detuning."""
        omega = 2 * np.pi
        n_cycles = 5

        gps_on_resonance = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=n_cycles, omega=omega, delta=0.0
        )
        gps_detuned = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=n_cycles, omega=omega, delta=0.5
        )

        phase_on_res = gps_on_resonance.differential_phase()
        phase_detuned = gps_detuned.differential_phase()

        # Phases should be different
        self.assertNotAlmostEqual(phase_on_res, phase_detuned, places=3)

    def test_scaling_with_cycles(self):
        """Test phase accumulation scales with number of cycles."""
        omega = 2 * np.pi
        delta = 0.1

        gps_1 = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=1, omega=omega, delta=delta
        )
        gps_5 = GlobalPhaseSpectroscopySequence(
            self.system, n_cycles=5, omega=omega, delta=delta
        )

        phase_1 = gps_1.differential_phase()
        phase_5 = gps_5.differential_phase()

        # Phase should roughly scale with time (and thus cycles)
        # Not exactly linear due to Rabi dynamics, but should grow
        self.assertGreater(np.abs(phase_5), np.abs(phase_1) * 0.5)


if __name__ == '__main__':
    unittest.main()
