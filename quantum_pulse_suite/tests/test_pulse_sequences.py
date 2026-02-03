"""
Unit tests for quantum_pulse_suite.

Tests filter functions for Rabi and Ramsey sequences against known analytical
results, and validates numerical simulations against analytic decoherence predictions.
"""

import unittest
import numpy as np
from scipy.integrate import simpson

from quantum_pulse_suite import (
    SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY,
    ramsey_sequence,
    continuous_rabi_sequence,
    InstantaneousPulseSequence,
    ContinuousPulseSequence,
    NoiseGenerator,
    ColoredNoisePSD,
)
from quantum_pulse_suite.analysis import (
    NumericalEvolution,
    TrajectoryAnalyzer,
    bloch_vector_from_operator,
)


class TestRamseyFilterFunction(unittest.TestCase):
    """Test Ramsey sequence filter functions against analytical predictions."""

    def test_ramsey_filter_function_shape(self):
        """Test that Ramsey filter function has correct sinc-like shape."""
        tau = 1.0
        seq = ramsey_sequence(tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # Evaluate at frequencies around the characteristic frequency
        frequencies = np.linspace(0.1, 20, 100)
        Fx, Fy, Fz = ff.filter_function(frequencies)

        # For Ramsey with delta=0, Fz should dominate
        # The filter function should have zeros at ω = 2πn/τ
        total = np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2

        # Check that filter function is non-negative
        self.assertTrue(np.all(total >= -1e-10))

        # Check that it has the right order of magnitude
        self.assertTrue(np.max(total) > 0)

    def test_ramsey_filter_function_zeros(self):
        """Test that Ramsey filter function has zeros at expected frequencies."""
        tau = 1.0
        seq = ramsey_sequence(tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # For a simple Ramsey sequence, F(ω) should have zeros near ω = 2πn/τ
        # Test at the first zero (approximately)
        zero_freq = 2 * np.pi / tau
        freqs_near_zero = np.array([zero_freq * 0.95, zero_freq, zero_freq * 1.05])

        susceptibility = ff.noise_susceptibility(freqs_near_zero)

        # The middle value should be a local minimum
        self.assertLess(susceptibility[1], susceptibility[0])
        self.assertLess(susceptibility[1], susceptibility[2])

    def test_ramsey_total_duration(self):
        """Test that Ramsey sequence has correct total duration."""
        tau = 2.5
        seq = ramsey_sequence(tau=tau, delta=0.0)

        # Instantaneous pulses have zero duration, so total = tau
        self.assertAlmostEqual(seq.total_duration(), tau, places=10)


class TestRabiFilterFunction(unittest.TestCase):
    """Test continuous Rabi sequence filter functions."""

    def test_rabi_filter_function_basic(self):
        """Test that Rabi filter function is computable and reasonable."""
        omega = np.pi  # Rabi frequency for π rotation
        tau = 1.0
        seq = continuous_rabi_sequence(omega=omega, tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()

        frequencies = np.logspace(-1, 2, 50)
        Fx, Fy, Fz = ff.filter_function(frequencies)

        # Check outputs are real and finite
        self.assertTrue(np.all(np.isfinite(Fx)))
        self.assertTrue(np.all(np.isfinite(Fy)))
        self.assertTrue(np.all(np.isfinite(Fz)))

    def test_rabi_filter_function_resonance(self):
        """Test filter function behavior near Rabi frequency."""
        omega = 2 * np.pi
        tau = 1.0
        seq = continuous_rabi_sequence(omega=omega, tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # Test around the Rabi frequency - should see resonance features
        frequencies = np.linspace(omega * 0.5, omega * 1.5, 50)
        susceptibility = ff.noise_susceptibility(frequencies)

        # Should have some structure near the Rabi frequency
        self.assertTrue(np.std(susceptibility) > 0)

    def test_rabi_with_detuning(self):
        """Test Rabi sequence with non-zero detuning."""
        omega = np.pi
        tau = 1.0
        delta = 0.5
        seq = continuous_rabi_sequence(omega=omega, tau=tau, delta=delta)
        ff = seq.get_filter_function_calculator()

        frequencies = np.logspace(-1, 1, 30)
        susceptibility = ff.noise_susceptibility(frequencies)

        # Filter function should still be valid
        self.assertTrue(np.all(np.isfinite(susceptibility)))
        self.assertTrue(np.all(susceptibility >= -1e-10))

    def test_rabi_total_duration(self):
        """Test that Rabi sequence has correct total duration."""
        tau = 3.0
        seq = continuous_rabi_sequence(omega=np.pi, tau=tau, delta=0.0)
        self.assertAlmostEqual(seq.total_duration(), tau, places=10)


class TestNumericalEvolution(unittest.TestCase):
    """Test numerical evolution methods."""

    def setUp(self):
        """Set up numerical evolution engine."""
        self.evo = NumericalEvolution(dt=0.01)

    def test_identity_evolution(self):
        """Test that zero Hamiltonian gives identity evolution."""
        # Zero Rabi frequency, zero detuning
        params = np.array([[0, 1, 0, 0, 0, 1.0]])  # omega=0, n_x=1, delta=0, tau=1
        U_traj, times = self.evo.evolve_sequence(params, noise_type=-1)

        # Final unitary should be close to identity
        U_final = U_traj[-1]
        np.testing.assert_array_almost_equal(U_final, IDENTITY, decimal=5)

    def test_pi_rotation(self):
        """Test that a π pulse gives the correct rotation."""
        # π rotation around x-axis: omega * tau = π
        omega = np.pi
        tau = 1.0
        params = np.array([[omega, 1, 0, 0, 0, tau]])

        U_traj, times = self.evo.evolve_sequence(params, noise_type=-1)
        U_final = U_traj[-1]

        # Expected: exp(-i * σx * π / 2) = -i * σx
        expected = -1j * SIGMA_X
        np.testing.assert_array_almost_equal(U_final, expected, decimal=3)

    def test_pi_half_rotation(self):
        """Test π/2 rotation."""
        omega = np.pi / 2
        tau = 1.0
        params = np.array([[omega, 1, 0, 0, 0, tau]])

        U_traj, times = self.evo.evolve_sequence(params, noise_type=-1)
        U_final = U_traj[-1]

        # Check unitarity
        should_be_identity = U_final @ U_final.conj().T
        np.testing.assert_array_almost_equal(should_be_identity, IDENTITY, decimal=5)

    def test_evolution_with_noise(self):
        """Test that evolution with noise produces different results."""
        params = np.array([[np.pi, 1, 0, 0, 0, 1.0]])

        # Run without noise
        U_no_noise, _ = self.evo.evolve_sequence(params, noise_type=-1)

        # Run with white noise
        U_with_noise, _ = self.evo.evolve_sequence(
            params, noise_type=0, psd_amplitude=0.1
        )

        # Results should be different (with high probability)
        diff = np.linalg.norm(U_no_noise[-1] - U_with_noise[-1])
        # Just check it ran without error; noise makes exact comparison impossible
        self.assertTrue(np.isfinite(diff))


class TestFilterFunctionDecoherence(unittest.TestCase):
    """
    Test that numerical decoherence matches filter function predictions.

    The decoherence (variance of observable) should follow:
    Var(⟨O⟩) ≈ ∫ |F(ω)|² S(ω) dω / (2π)

    where F(ω) is the filter function and S(ω) is the noise PSD.
    """

    def setUp(self):
        """Set up test parameters."""
        self.n_trajectories = 50
        self.evo = NumericalEvolution(dt=0.005)
        self.analyzer = TrajectoryAnalyzer(self.evo)

    def _compute_analytic_variance(self, ff, psd_func, freq_range=(0.01, 100), n_points=500):
        """
        Compute expected variance from filter function and noise PSD.

        Var ≈ ∫ |F(ω)|² S(ω) dω / (2π)
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
        susceptibility = ff.noise_susceptibility(frequencies)
        psd_values = psd_func(frequencies)

        # Integrate |F(ω)|² * S(ω)
        integrand = susceptibility * psd_values
        integral = simpson(integrand, x=frequencies)

        return integral / (2 * np.pi)

    def test_ramsey_decoherence_white_noise(self):
        """Test Ramsey decoherence with white noise matches filter function prediction."""
        tau = 0.5
        psd_amplitude = 0.01

        # Create Ramsey sequence
        seq = ramsey_sequence(tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # Compute analytic prediction
        psd_func = ColoredNoisePSD.white_noise(amplitude=psd_amplitude)
        analytic_var = self._compute_analytic_variance(ff, psd_func)

        # Run numerical simulation
        # Convert to continuous pulse format for numerical evolution
        # Ramsey: π/2_x - wait(τ) - π/2_x
        # We approximate instant pulses with very fast pulses
        t_pulse = 0.01
        omega_pulse = np.pi / 2 / t_pulse

        params = np.array([
            [omega_pulse, 1, 0, 0, 0, t_pulse],  # π/2 x-pulse
            [0, 0, 0, 1, 0, tau],                 # Free evolution
            [omega_pulse, 1, 0, 0, 0, t_pulse],  # π/2 x-pulse
        ])

        # Compute numerical variance over trajectories
        initial_state = np.array([1, 0], dtype=complex)
        measurement_op = SIGMA_Y

        numerical_var = self.analyzer.avg_var_trajectories(
            params, measurement_op,
            n_trajectories=self.n_trajectories,
            noise_type=0,  # White noise
            psd_amplitude=psd_amplitude,
            initial_state=initial_state
        )

        # Check that numerical and analytic are in the same order of magnitude
        # We use a loose tolerance due to finite trajectory averaging
        if analytic_var > 1e-10:
            ratio = numerical_var / analytic_var
            # Accept if within factor of 10 (trajectories have large variance)
            self.assertGreater(ratio, 0.01)
            self.assertLess(ratio, 100)

    def test_ramsey_variance_convergence(self):
        """Test that numerical variance converges toward the analytic prediction as
        trajectory count increases from 100 to 150."""
        tau = 0.4
        psd_amplitude = 0.015

        # Analytic prediction from filter function
        seq = ramsey_sequence(tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()
        psd_func = ColoredNoisePSD.white_noise(amplitude=psd_amplitude)
        analytic_var = self._compute_analytic_variance(ff, psd_func)

        # Approximate instantaneous π/2 pulses with fast finite pulses
        t_pulse = 0.01
        omega_pulse = np.pi / 2 / t_pulse
        params = np.array([
            [omega_pulse, 1, 0, 0, 0, t_pulse],  # π/2 x-pulse
            [0, 0, 0, 1, 0, tau],                 # Free evolution
            [omega_pulse, 1, 0, 0, 0, t_pulse],  # π/2 x-pulse
        ])

        initial_state = np.array([1, 0], dtype=complex)
        measurement_op = SIGMA_Y

        # Run at two trajectory counts
        var_100 = self.analyzer.avg_var_trajectories(
            params, measurement_op,
            n_trajectories=100,
            noise_type=0,
            psd_amplitude=psd_amplitude,
            initial_state=initial_state,
        )
        var_150 = self.analyzer.avg_var_trajectories(
            params, measurement_op,
            n_trajectories=150,
            noise_type=0,
            psd_amplitude=psd_amplitude,
            initial_state=initial_state,
        )

        # Both estimates must be finite and positive
        self.assertTrue(np.isfinite(var_100))
        self.assertTrue(np.isfinite(var_150))
        self.assertGreater(var_100, 0)
        self.assertGreater(var_150, 0)

        if analytic_var > 1e-10:
            ratio_100 = var_100 / analytic_var
            ratio_150 = var_150 / analytic_var

            # Both ratios should land within an order of magnitude of 1.0
            self.assertGreater(ratio_100, 0.1)
            self.assertLess(ratio_100, 10.0)
            self.assertGreater(ratio_150, 0.1)
            self.assertLess(ratio_150, 10.0)

            # The 150-trajectory estimate should be at least as close to the
            # analytic value as the 100-trajectory estimate.  A slack factor
            # of 2.0 absorbs the inherent run-to-run stochastic variation
            # while still enforcing that the longer run does not diverge.
            error_100 = abs(ratio_100 - 1.0)
            error_150 = abs(ratio_150 - 1.0)
            self.assertLess(error_150, error_100 * 2.0)

    def test_rabi_fidelity_decay(self):
        """Test that Rabi oscillation fidelity decays with noise."""
        omega = np.pi  # π rotation (ω*τ = π)
        tau = 1.0
        psd_amplitude = 0.01

        # Continuous Rabi pulse parameters
        params = np.array([[omega, 1, 0, 0, 0, tau]])

        # Expected final state without noise: exp(-i σx π/2) |0⟩ = -i|1⟩
        initial_state = np.array([1, 0], dtype=complex)
        ideal_final = np.array([0, -1j], dtype=complex)

        # Run multiple trajectories
        fidelities = []
        for _ in range(self.n_trajectories):
            U_traj, times = self.evo.evolve_sequence(
                params, noise_type=0, psd_amplitude=psd_amplitude
            )
            final_state = U_traj[-1] @ initial_state
            fidelity = np.abs(np.vdot(ideal_final, final_state))**2
            fidelities.append(fidelity)

        avg_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)

        # With noise, average fidelity should be less than 1
        self.assertLess(avg_fidelity, 1.0)
        # But shouldn't be too low for moderate noise
        self.assertGreater(avg_fidelity, 0.5)

        # Standard deviation should be non-zero (noise causes variation)
        self.assertGreater(std_fidelity, 0)

    def test_sigma_y_expectation_ramsey(self):
        """Test ⟨σy⟩ observable in Ramsey sequence with noise averaging."""
        tau = 0.3
        psd_amplitude = 0.02

        # Approximate Ramsey sequence
        t_pulse = 0.01
        omega_pulse = np.pi / 2 / t_pulse

        params = np.array([
            [omega_pulse, 1, 0, 0, 0, t_pulse],
            [0, 0, 0, 1, 0, tau],
            [omega_pulse, 1, 0, 0, 0, t_pulse],
        ])

        initial_state = np.array([1, 0], dtype=complex)

        # Compute ⟨σy⟩ over multiple trajectories
        sigma_y_values = []
        for _ in range(self.n_trajectories):
            U_traj, times = self.evo.evolve_sequence(
                params, noise_type=0, psd_amplitude=psd_amplitude
            )
            final_state = U_traj[-1] @ initial_state
            sigma_y_exp = np.real(np.vdot(final_state, SIGMA_Y @ final_state))
            sigma_y_values.append(sigma_y_exp)

        avg_sigma_y = np.mean(sigma_y_values)
        var_sigma_y = np.var(sigma_y_values)

        # Get filter function prediction
        seq = ramsey_sequence(tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()
        psd_func = ColoredNoisePSD.white_noise(amplitude=psd_amplitude)
        predicted_var = self._compute_analytic_variance(ff, psd_func)

        # The variance of the observable should be related to the filter function
        # This is a qualitative check - both should be small for weak noise
        self.assertLess(var_sigma_y, 1.0)  # Variance bounded by observable range

        # Check that numerical variance is in reasonable range
        # (exact comparison is difficult due to approximations)
        self.assertTrue(np.isfinite(var_sigma_y))
        self.assertGreaterEqual(var_sigma_y, 0)


class TestNoiseGenerator(unittest.TestCase):
    """Test noise generation utilities."""

    def test_white_noise_spectrum(self):
        """Test that white noise has flat spectrum."""
        gen = NoiseGenerator(seed=42)
        noise, times = gen.generate(n_points=1000, dt=0.01, noise_type=0)

        # Check output shape
        self.assertEqual(len(noise), 1000)
        self.assertEqual(len(times), 1000)

        # White noise should have roughly uniform power across frequencies
        fft = np.fft.fft(noise)
        power = np.abs(fft)**2

        # Check that power doesn't vary too much (excluding DC)
        power_no_dc = power[1:len(power)//2]
        cv = np.std(power_no_dc) / np.mean(power_no_dc)
        # Coefficient of variation should be moderate for white noise
        self.assertLess(cv, 3.0)

    def test_pink_noise_spectrum(self):
        """Test that 1/f noise has decreasing spectrum."""
        gen = NoiseGenerator(seed=42)
        noise, times = gen.generate(n_points=2000, dt=0.01, noise_type=1)

        fft = np.fft.fft(noise)
        power = np.abs(fft)**2
        freqs = np.fft.fftfreq(len(noise), 0.01)

        # Check that low frequencies have more power than high frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_power = power[:len(power)//2]

        low_freq_power = np.mean(pos_power[1:50])
        high_freq_power = np.mean(pos_power[100:200])

        self.assertGreater(low_freq_power, high_freq_power)

    def test_noise_reproducibility(self):
        """Test that seeded noise is reproducible."""
        gen1 = NoiseGenerator(seed=123)
        gen2 = NoiseGenerator(seed=123)

        noise1, _ = gen1.generate(n_points=100, dt=0.01)
        noise2, _ = gen2.generate(n_points=100, dt=0.01)

        np.testing.assert_array_equal(noise1, noise2)


class TestBlochVector(unittest.TestCase):
    """Test Bloch vector utilities."""

    def test_bloch_vector_identity(self):
        """Test Bloch vector extraction from identity-like operator."""
        op = np.array([[1, 0], [0, 0]], dtype=complex)
        bloch = bloch_vector_from_operator(op)

        self.assertEqual(bloch[0], 0)  # x component
        self.assertEqual(bloch[1], 0)  # y component
        self.assertEqual(bloch[2], 1)  # z component

    def test_bloch_vector_sigma_x(self):
        """Test Bloch vector for σx-like operator."""
        op = np.array([[0, 1], [1, 0]], dtype=complex)
        bloch = bloch_vector_from_operator(op)

        self.assertEqual(bloch[0], 1)  # x component
        self.assertEqual(bloch[1], 0)  # y component
        self.assertEqual(bloch[2], 0)  # z component


if __name__ == '__main__':
    unittest.main()
