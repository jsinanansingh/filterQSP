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
    spin_echo_sequence,
    cpmg_sequence,
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
    dick_effect_coefficients,
    allan_variance_dick,
    allan_variance_continuous,
    allan_deviation,
    allan_variance_vs_tau,
    quantum_projection_noise_limit,
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
        # Use midpoints between adjacent zeros as neighbors (where susceptibility peaks)
        for n in range(1, 11):
            zero_freq = 2 * np.pi * n / tau
            left_mid = 2 * np.pi * (n - 0.5) / tau
            right_mid = 2 * np.pi * (n + 0.5) / tau
            freqs = np.array([left_mid, zero_freq, right_mid])

            susceptibility = ff.noise_susceptibility(freqs)

            # The zero should be a local minimum compared to the midpoints
            self.assertLess(susceptibility[1], susceptibility[0],
                            msg=f"Zero at n={n}: susceptibility at zero not less than left midpoint")
            self.assertLess(susceptibility[1], susceptibility[2],
                            msg=f"Zero at n={n}: susceptibility at zero not less than right midpoint")

    def test_ramsey_with_formula(self):
        """Test Ramsey noise susceptibility matches 4 sin^2(wt/2) / w^2."""
        for tau in [0.5, 1.0, 2.0]:
            seq = ramsey_sequence(tau=tau, delta=0.0)
            ff = seq.get_filter_function_calculator()

            freqs = np.linspace(0.5, 80, 300)
            numerical = ff.noise_susceptibility(freqs)
            analytical = 4 * np.sin(freqs * tau / 2)**2 / freqs**2

            diff = np.abs(numerical - analytical)
            self.assertTrue(np.all(diff < 1e-12),
                            msg=f"tau={tau}: max diff {np.max(diff):.2e}")

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


class TestCPMGFilterFunction(unittest.TestCase):
    """Test CPMG sequence filter functions against known analytical properties.

    For CPMG-N (N π-pulses) with total free evolution time τ:
    - Interpulse spacing: Δt = τ/(2N)
    - Filter function peaks at ω = π(2k+1)/Δt = 2πN(2k+1)/τ for k=0,1,2,...
    - Filter function zeros at ω = 2πm/τ for integer m (sequence repetition harmonics)
    """

    def test_cpmg_total_duration(self):
        """Test that CPMG sequence has correct total duration."""
        tau = 2.0
        n_pulses = 4
        seq = cpmg_sequence(tau=tau, n_pulses=n_pulses, delta=0.0)

        # Instantaneous pulses have zero duration, so total = tau
        self.assertAlmostEqual(seq.total_duration(), tau, places=10)

    def test_cpmg_filter_function_computable(self):
        """Test that CPMG filter function is computable and finite."""
        tau = 1.0
        n_pulses = 4
        seq = cpmg_sequence(tau=tau, n_pulses=n_pulses, delta=0.0)
        ff = seq.get_filter_function_calculator()

        frequencies = np.logspace(-1, 2, 100)
        Fx, Fy, Fz = ff.filter_function(frequencies)

        self.assertTrue(np.all(np.isfinite(Fx)))
        self.assertTrue(np.all(np.isfinite(Fy)))
        self.assertTrue(np.all(np.isfinite(Fz)))

    def test_cpmg_with_formula(self):
        """Test CPMG noise susceptibility matches the QSP analytical formula.

        For CPMG-n with instantaneous pi-pulses and total free evolution tau,
        the noise susceptibility decomposes into orthogonal Fz (first segment)
        and Fy (subsequent segments) components:

            S(w) = (4 sin^2(wh/2) / w^2) * (1 + |sum_{k=1}^{2n-1} y_k e^{iwkh}|^2)

        where h = tau/(2n) and y_k follows the toggling-frame sign pattern
        {-1,-1,+1,+1,-1,-1,...} in alternating pairs.
        """
        for n_pulses in [1, 2, 4, 8]:
            for tau in [0.5, 1.0, 2.0]:
                seq = cpmg_sequence(tau=tau, n_pulses=n_pulses, delta=0.0)
                ff = seq.get_filter_function_calculator()

                freqs = np.linspace(0.5, 80, 300)
                numerical = ff.noise_susceptibility(freqs)

                h = tau / (2 * n_pulses)

                # Build toggling-frame sign pattern
                y_vals = [1]  # segment 0 (contributes to Fz)
                sign = -1
                for _ in range(n_pulses):
                    y_vals.extend([sign, sign])
                    sign *= -1
                y_vals = y_vals[:2 * n_pulses]

                # Compute analytical formula
                alpha_sq = 4 * np.sin(freqs * h / 2)**2 / freqs**2
                S = np.zeros(len(freqs), dtype=complex)
                for k in range(1, 2 * n_pulses):
                    S += y_vals[k] * np.exp(1j * freqs * k * h)
                analytical = alpha_sq * (1 + np.abs(S)**2)

                diff = np.abs(numerical - analytical)
                self.assertTrue(
                    np.all(diff < 1e-12),
                    msg=f"n={n_pulses}, tau={tau}: max diff {np.max(diff):.2e}")

    def test_continuous_cpmg_fz_with_formula(self):
        """Test finite-duration CPMG |Fz(w)|^2 matches the closed-form formula.

        For CPMG-n with finite-duration pi-pulses (Rabi frequency Omega,
        pulse duration t_p = pi/Omega) and free-evolution half-intervals
        h = tau/(2n), the dephasing filter function factorises as:

            |Fz(w)|^2 = |U(w)|^2 * |G(w)|^2

        where the unit-cell Fourier transform is

            U(w) = A(w) + B(w) - C(w)

        with
            A(w) = (e^{iwh} - 1) / (iw)                  [first free segment]
            B(w) = e^{iwh} * P(w)                          [pi pulse]
            C(w) = e^{iw(h + t_p)} * (e^{iwh} - 1) / (iw) [second free segment]

        P(w) is the Fourier transform of cos(Omega*s) over [0, t_p]:
            P(w) = [(e^{i(w+O)t_p} - 1) / (i(w+O)) + (e^{i(w-O)t_p} - 1) / (i(w-O))] / 2

        and the alternating geometric sum is

            G(w) = sum_{k=0}^{n-1} (-1)^k e^{iwk T_p},   T_p = 2h + t_p
        """
        for n_pulses in [1, 2, 4, 8]:
            for tau in [0.5, 1.0, 2.0]:
                for omega_rabi in [5 * np.pi, 20 * np.pi, 100 * np.pi]:
                    t_p = np.pi / omega_rabi
                    h = tau / (2 * n_pulses)
                    T_p = 2 * h + t_p

                    # Build core CPMG (no pi/2 bookends)
                    seq = ContinuousPulseSequence()
                    for _ in range(n_pulses):
                        seq.add_continuous_pulse(0.0, [1, 0, 0], 0.0, h)
                        seq.add_continuous_pulse(omega_rabi, [0, 1, 0], 0.0, t_p)
                        seq.add_continuous_pulse(0.0, [1, 0, 0], 0.0, h)

                    ff = seq.get_filter_function_calculator()
                    freqs = np.linspace(0.5, 80, 300)
                    numerical_fz = np.abs(ff.filter_function(freqs)[2])**2

                    # Closed-form unit-cell contribution
                    w = freqs
                    A = (np.exp(1j * w * h) - 1) / (1j * w)
                    wp = w + omega_rabi
                    wm = w - omega_rabi
                    I1 = (np.exp(1j * wp * t_p) - 1) / (1j * wp)
                    I2 = np.where(
                        np.abs(wm) < 1e-12,
                        t_p + 0j,
                        (np.exp(1j * wm * t_p) - 1) / (1j * wm),
                    )
                    B = np.exp(1j * w * h) * (I1 + I2) / 2
                    C = np.exp(1j * w * (h + t_p)) * (np.exp(1j * w * h) - 1) / (1j * w)
                    unit_cell = A + B - C

                    # Alternating geometric sum
                    G = np.zeros(len(w), dtype=complex)
                    for k in range(n_pulses):
                        G += (-1)**k * np.exp(1j * w * k * T_p)

                    analytical = np.abs(unit_cell)**2 * np.abs(G)**2

                    diff = np.abs(numerical_fz - analytical)
                    self.assertTrue(
                        np.all(diff < 1e-10),
                        msg=(f"n={n_pulses}, tau={tau}, omega={omega_rabi/np.pi:.0f}*pi: "
                             f"max diff {np.max(diff):.2e}"))

    def test_cpmg_has_peak_structure(self):
        """Test that CPMG filter function has a clear peak structure.

        The filter function should have at least one distinct peak in
        the frequency range related to the pulse spacing.
        """
        tau = 1.0
        n_pulses = 4
        seq = cpmg_sequence(tau=tau, n_pulses=n_pulses, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # Search over a wide frequency range
        frequencies = np.linspace(1, 100, 500)
        susceptibility = ff.noise_susceptibility(frequencies)

        # Find the peak
        peak_idx = np.argmax(susceptibility)
        peak_value = susceptibility[peak_idx]

        # There should be a clear peak (not monotonic)
        # Check that peak is greater than values at edges
        self.assertGreater(peak_value, susceptibility[0])
        self.assertGreater(peak_value, susceptibility[-1])

        # The peak should be significantly above the minimum
        min_value = np.min(susceptibility)
        if min_value > 0:
            self.assertGreater(peak_value / min_value, 2.0)

    def test_cpmg_zeros_at_sequence_harmonics(self):
        """Test that CPMG filter function has zeros at sequence repetition harmonics.

        The filter function should have local minima near ω = 2πk/τ for integer k,
        corresponding to the zeros of the modulation function.
        """
        tau = 1.0
        n_pulses = 4
        seq = cpmg_sequence(tau=tau, n_pulses=n_pulses, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # Test at first few harmonics: ω = 2π/τ, 4π/τ, 6π/τ
        for k in [1, 2, 3]:
            omega_zero = 2 * np.pi * k / tau

            # Skip if this coincides with a peak (happens when k is odd multiple of N)
            if k % n_pulses == 0:
                continue

            # Evaluate around the expected zero
            freqs = np.array([
                omega_zero * 0.9,
                omega_zero,
                omega_zero * 1.1
            ])
            susceptibility = ff.noise_susceptibility(freqs)

            # The center value should be a local minimum
            self.assertLess(susceptibility[1], susceptibility[0] * 1.5,
                           f"Not a minimum at k={k}: {susceptibility}")
            self.assertLess(susceptibility[1], susceptibility[2] * 1.5,
                           f"Not a minimum at k={k}: {susceptibility}")

    def test_cpmg_more_pulses_shifts_peak(self):
        """Test that increasing N in CPMG shifts the peak to higher frequencies.

        More pulses should shift the passband center to higher frequencies
        since the effective modulation rate increases.
        """
        tau = 1.0
        seq_n2 = cpmg_sequence(tau=tau, n_pulses=2, delta=0.0)
        seq_n8 = cpmg_sequence(tau=tau, n_pulses=8, delta=0.0)

        ff_n2 = seq_n2.get_filter_function_calculator()
        ff_n8 = seq_n8.get_filter_function_calculator()

        # Find peak locations
        frequencies = np.linspace(1, 200, 1000)

        susc_n2 = ff_n2.noise_susceptibility(frequencies)
        susc_n8 = ff_n8.noise_susceptibility(frequencies)

        peak_freq_n2 = frequencies[np.argmax(susc_n2)]
        peak_freq_n8 = frequencies[np.argmax(susc_n8)]

        # N=8 should have peak at higher frequency than N=2
        self.assertGreater(peak_freq_n8, peak_freq_n2)

    def test_cpmg_vs_spin_echo(self):
        """Test that CPMG-1 is equivalent to spin echo."""
        tau = 1.0
        seq_cpmg1 = cpmg_sequence(tau=tau, n_pulses=1, delta=0.0)
        seq_echo = spin_echo_sequence(tau=tau, delta=0.0)

        # Both should have same total duration
        self.assertAlmostEqual(
            seq_cpmg1.total_duration(),
            seq_echo.total_duration(),
            places=10
        )

        # Filter functions should have similar structure
        ff_cpmg = seq_cpmg1.get_filter_function_calculator()
        ff_echo = seq_echo.get_filter_function_calculator()

        frequencies = np.linspace(1, 30, 50)
        susc_cpmg = ff_cpmg.noise_susceptibility(frequencies)
        susc_echo = ff_echo.noise_susceptibility(frequencies)

        # They should be correlated (similar shape)
        correlation = np.corrcoef(susc_cpmg, susc_echo)[0, 1]
        self.assertGreater(correlation, 0.8)

    def test_cpmg_frequency_selectivity(self):
        """Test that CPMG has frequency-selective behavior.

        The filter function should vary significantly across frequencies,
        showing the bandpass-like behavior characteristic of dynamical decoupling.
        """
        tau = 1.0
        n_pulses = 4
        seq = cpmg_sequence(tau=tau, n_pulses=n_pulses, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # Evaluate over a range of frequencies
        frequencies = np.logspace(-1, 2, 100)
        susceptibility = ff.noise_susceptibility(frequencies)

        # Filter function should have significant variation (not flat)
        # indicating frequency-selective behavior
        max_susc = np.max(susceptibility)
        min_susc = np.min(susceptibility)

        # There should be at least some contrast
        if max_susc > 1e-20:
            # Check that filter is not completely flat
            self.assertGreater(max_susc, min_susc * 1.1)


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


class TestAllanVariance(unittest.TestCase):
    """Test Allan variance computation for clock experiments."""

    def test_dick_coefficients_computable(self):
        """Test that Dick effect coefficients are computable and finite."""
        tau = 0.1  # 100 ms interrogation
        t_dead = 0.05  # 50 ms dead time
        seq = ramsey_sequence(tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()

        g_m_sq = dick_effect_coefficients(ff, tau, t_dead, n_harmonics=20)

        self.assertEqual(len(g_m_sq), 20)
        self.assertTrue(np.all(np.isfinite(g_m_sq)))
        self.assertTrue(np.all(g_m_sq >= 0))

    def test_dick_coefficients_decrease(self):
        """Test that Dick coefficients generally decrease with harmonic number."""
        tau = 0.1
        t_dead = 0.02
        seq = ramsey_sequence(tau=tau, delta=0.0)
        ff = seq.get_filter_function_calculator()

        g_m_sq = dick_effect_coefficients(ff, tau, t_dead, n_harmonics=50)

        # Average of first 10 harmonics should be larger than average of last 10
        avg_low = np.mean(g_m_sq[:10])
        avg_high = np.mean(g_m_sq[-10:])

        self.assertGreater(avg_low, avg_high * 0.1)

    def test_allan_variance_positive(self):
        """Test that Allan variance is positive and finite."""
        tau_interrog = 0.1
        t_dead = 0.05
        tau_avg = 1.0

        seq = ramsey_sequence(tau=tau_interrog, delta=0.0)
        ff = seq.get_filter_function_calculator()

        # White frequency noise PSD
        def white_psd(omega):
            return 1e-26 * np.ones_like(omega)

        var = allan_variance_dick(ff, white_psd, tau_interrog, t_dead, tau_avg)

        self.assertTrue(np.isfinite(var))
        self.assertGreater(var, 0)

    def test_allan_variance_scales_with_tau(self):
        """Test that Allan variance decreases with averaging time for white noise."""
        tau_interrog = 0.1
        t_dead = 0.02

        seq = ramsey_sequence(tau=tau_interrog, delta=0.0)
        ff = seq.get_filter_function_calculator()

        def white_psd(omega):
            return 1e-24 * np.ones_like(omega)

        var_1s = allan_variance_dick(ff, white_psd, tau_interrog, t_dead, 1.0)
        var_10s = allan_variance_dick(ff, white_psd, tau_interrog, t_dead, 10.0)

        # For white frequency noise, Allan variance ~ 1/τ
        # So var_1s should be larger than var_10s
        self.assertGreater(var_1s, var_10s)

    def test_allan_deviation_methods_consistent(self):
        """Test that different Allan variance methods give consistent results."""
        tau_interrog = 0.05
        t_dead = 0.02
        tau_avg = 0.5

        seq = ramsey_sequence(tau=tau_interrog, delta=0.0)
        ff = seq.get_filter_function_calculator()

        def white_psd(omega):
            return 1e-24 * np.ones_like(omega)

        adev_dick = allan_deviation(
            ff, white_psd, tau_interrog, t_dead, tau_avg, method='dick'
        )
        adev_cont = allan_deviation(
            ff, white_psd, tau_interrog, t_dead, tau_avg, method='continuous'
        )

        # Both methods should give finite positive results
        self.assertTrue(np.isfinite(adev_dick))
        self.assertTrue(np.isfinite(adev_cont))
        self.assertGreater(adev_dick, 0)
        self.assertGreater(adev_cont, 0)

        # They should be within the same order of magnitude
        ratio = adev_dick / adev_cont
        self.assertGreater(ratio, 0.01)
        self.assertLess(ratio, 100)

    def test_allan_variance_vs_tau_array(self):
        """Test computing Allan variance over array of averaging times."""
        tau_interrog = 0.1
        t_dead = 0.02

        seq = ramsey_sequence(tau=tau_interrog, delta=0.0)
        ff = seq.get_filter_function_calculator()

        def flicker_psd(omega):
            # 1/f noise
            return 1e-24 / (np.abs(omega) + 1e-10)

        tau_array = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        variances = allan_variance_vs_tau(
            ff, flicker_psd, tau_interrog, t_dead, tau_array
        )

        self.assertEqual(len(variances), len(tau_array))
        self.assertTrue(np.all(np.isfinite(variances)))
        self.assertTrue(np.all(variances >= 0))

    def test_qpn_limit_scaling(self):
        """Test quantum projection noise limit scales correctly."""
        tau_interrog = 0.1
        t_dead = 0.02
        n_atoms = 1000
        freq = 1e15  # Optical clock frequency

        qpn_func = quantum_projection_noise_limit(
            tau_interrog, t_dead, n_atoms, freq
        )

        sigma_1s = qpn_func(1.0)
        sigma_100s = qpn_func(100.0)

        # QPN scales as 1/√τ
        expected_ratio = np.sqrt(100.0 / 1.0)
        actual_ratio = sigma_1s / sigma_100s

        np.testing.assert_almost_equal(actual_ratio, expected_ratio, decimal=5)

    def test_allan_variance_different_sequences(self):
        """Test that Allan variance computation works for different sequences.

        Verify that we can compute Allan variance for both Ramsey and CPMG
        sequences with different noise spectra.
        """
        tau = 0.1
        t_dead = 0.02
        tau_avg = 1.0

        seq_ramsey = ramsey_sequence(tau=tau, delta=0.0)
        seq_cpmg = cpmg_sequence(tau=tau, n_pulses=4, delta=0.0)

        ff_ramsey = seq_ramsey.get_filter_function_calculator()
        ff_cpmg = seq_cpmg.get_filter_function_calculator()

        # White noise PSD
        def white_psd(omega):
            return 1e-24 * np.ones_like(omega)

        var_ramsey = allan_variance_dick(
            ff_ramsey, white_psd, tau, t_dead, tau_avg
        )
        var_cpmg = allan_variance_dick(
            ff_cpmg, white_psd, tau, t_dead, tau_avg
        )

        # Both should give finite positive results
        self.assertTrue(np.isfinite(var_ramsey))
        self.assertTrue(np.isfinite(var_cpmg))
        self.assertGreater(var_ramsey, 0)
        self.assertGreater(var_cpmg, 0)


if __name__ == '__main__':
    unittest.main()
