"""
Unit tests for FFT-based filter function computation.

Tests that the FFT matrix filter function produces the correct noise
susceptibility. For instantaneous sequences, the FFT is compared to both
the analytic Bloch-vector filter function (Ramsey) and the known switching-
function integral (spin echo, CPMG). For continuous sequences, the FFT is
compared against direct numerical quadrature of the toggling-frame Hamiltonian.
"""

import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

from quantum_pulse_suite import (
    SIGMA_X, SIGMA_Y, SIGMA_Z,
    ramsey_sequence,
    spin_echo_sequence,
    cpmg_sequence,
    continuous_ramsey_sequence,
    continuous_cpmg_sequence,
    fft_filter_function,
    noise_susceptibility_from_matrix,
    bloch_components_from_matrix,
)


def _switching_function_susceptibility(pulse_times, T, freqs):
    """Compute noise susceptibility from the switching function integral.

    For instantaneous pulse sequences with z-noise, the toggling-frame
    Hamiltonian is piecewise constant:  h(t) = s_j * (-sigma_y/2) where
    s_j = +1 or -1 alternates at each pi pulse.

    The noise susceptibility equals |integral s(t) exp(-iwt) dt|^2.

    Parameters
    ----------
    pulse_times : list of float
        Times of the pi-pulses (NOT including the initial/final pi/2 pulses).
    T : float
        Total free evolution time.
    freqs : np.ndarray
        Angular frequencies.

    Returns
    -------
    np.ndarray
        Noise susceptibility at each frequency.
    """
    # Build segment boundaries and signs
    boundaries = [0.0] + list(pulse_times) + [T]
    n_segs = len(boundaries) - 1

    suscept = np.zeros(len(freqs))
    for fi, w in enumerate(freqs):
        integral = 0.0 + 0.0j
        sign = 1.0
        for j in range(n_segs):
            t_start = boundaries[j]
            t_end = boundaries[j + 1]
            # integral of sign * exp(-iwt) from t_start to t_end
            if abs(w) < 1e-12:
                integral += sign * (t_end - t_start)
            else:
                integral += sign * (np.exp(-1j * w * t_end) -
                                     np.exp(-1j * w * t_start)) / (-1j * w)
            sign *= -1  # flip at each pi pulse
        suscept[fi] = np.abs(integral)**2

    return suscept


def _quadrature_susceptibility(seq, noise_hamiltonian, freqs):
    """Compute noise susceptibility via direct numerical quadrature."""
    elements = seq.elements
    d = noise_hamiltonian.shape[0]
    T = seq.total_duration()

    def unitary_at_t(t):
        U = np.eye(d, dtype=complex)
        cumtime = 0.0
        for elem in elements:
            dur = elem.duration()
            if dur == 0:
                if cumtime <= t:
                    U = elem.unitary() @ U
            else:
                if cumtime >= t:
                    break
                elif cumtime + dur <= t:
                    U = elem.unitary() @ U
                    cumtime += dur
                else:
                    dt_in = t - cumtime
                    H = elem.hamiltonian()
                    U = expm(-1j * H * dt_in) @ U
                    break
        return U

    sigmas = [SIGMA_X, SIGMA_Y, SIGMA_Z]

    suscept = np.zeros(len(freqs))
    for fi, w in enumerate(freqs):
        for sigma_k in sigmas:
            def integrand_re(t, sk=sigma_k, ww=w):
                U = unitary_at_t(t)
                h = U.conj().T @ noise_hamiltonian @ U
                hk = np.real(np.trace(sk @ h))
                return hk * np.cos(ww * t)

            def integrand_im(t, sk=sigma_k, ww=w):
                U = unitary_at_t(t)
                h = U.conj().T @ noise_hamiltonian @ U
                hk = np.real(np.trace(sk @ h))
                return -hk * np.sin(ww * t)

            re_part, _ = quad(integrand_re, 0, T, limit=200)
            im_part, _ = quad(integrand_im, 0, T, limit=200)
            suscept[fi] += re_part**2 + im_part**2

    return suscept


class TestFFTFilterFunctionQubit(unittest.TestCase):
    """Test FFT filter function against reference calculations."""

    def test_ramsey_vs_analytic(self):
        """Ramsey: FFT noise susceptibility matches analytic."""
        for tau in [1.0, 2.0, 2 * np.pi]:
            seq = ramsey_sequence(tau=tau, delta=0.0)
            seq.compute_polynomials()
            ff = seq.get_filter_function_calculator()

            H_noise = SIGMA_Z / 2
            freqs, F_mat = fft_filter_function(seq, H_noise, n_samples=8192)
            suscept_fft = noise_susceptibility_from_matrix(F_mat)
            suscept_ana = ff.noise_susceptibility(freqs)

            T = seq.total_duration()
            mask = (freqs > 2 * np.pi / T) & (freqs < 50.0)
            s_fft = suscept_fft[mask]
            s_ana = suscept_ana[mask]

            peak = max(np.max(s_fft), np.max(s_ana))
            sig = (s_ana > 1e-4 * peak) & (s_fft > 1e-4 * peak)
            if np.any(sig):
                rel_err = np.abs(s_fft[sig] - s_ana[sig]) / s_ana[sig]
                self.assertLess(
                    np.max(rel_err), 0.02,
                    msg=f"Ramsey(tau={tau:.2f}): max rel error {np.max(rel_err):.4f}"
                )

    def test_ramsey_with_detuning(self):
        """Ramsey with detuning: susceptibility is delta-independent for inst."""
        seq0 = ramsey_sequence(tau=2.0, delta=0.0)
        seq1 = ramsey_sequence(tau=2.0, delta=0.5)
        seq0.compute_polynomials()
        seq1.compute_polynomials()
        ff0 = seq0.get_filter_function_calculator()
        ff1 = seq1.get_filter_function_calculator()
        freqs = np.linspace(0.5, 50, 200)
        np.testing.assert_allclose(
            ff0.noise_susceptibility(freqs),
            ff1.noise_susceptibility(freqs),
            atol=1e-10,
            err_msg="Inst Ramsey susceptibility should be delta-independent"
        )

    def test_spin_echo_vs_switching(self):
        """Spin echo: FFT matches the switching function integral."""
        tau = 2.0
        seq = spin_echo_sequence(tau=tau, delta=0.0)
        H_noise = SIGMA_Z / 2
        freqs, F_mat = fft_filter_function(seq, H_noise, n_samples=8192)
        suscept_fft = noise_susceptibility_from_matrix(F_mat)

        # Spin echo: one pi pulse at T/2
        suscept_ref = _switching_function_susceptibility([tau / 2], tau, freqs)

        # Compare at significant frequencies
        mask = (freqs > 1.0) & (freqs < 200.0)
        sig = suscept_ref[mask] > 1e-4 * np.max(suscept_ref[mask])
        np.testing.assert_allclose(
            suscept_fft[mask][sig], suscept_ref[mask][sig], rtol=0.005,
            err_msg="SpinEcho FFT != switching function"
        )

    def test_cpmg_vs_switching(self):
        """CPMG-n: FFT matches the switching function integral."""
        tau = 2.0
        for n in [1, 2, 4]:
            seq = cpmg_sequence(tau=tau, n_pulses=n, delta=0.0)
            H_noise = SIGMA_Z / 2
            freqs, F_mat = fft_filter_function(seq, H_noise, n_samples=8192)
            suscept_fft = noise_susceptibility_from_matrix(F_mat)

            # CPMG-n: pi pulses at T/(2n), 3T/(2n), 5T/(2n), ...
            pulse_times = [(2 * k + 1) * tau / (2 * n) for k in range(n)]
            suscept_ref = _switching_function_susceptibility(pulse_times, tau, freqs)

            mask = (freqs > 1.0) & (freqs < 200.0)
            sig = suscept_ref[mask] > 1e-4 * np.max(suscept_ref[mask])
            np.testing.assert_allclose(
                suscept_fft[mask][sig], suscept_ref[mask][sig], rtol=0.005,
                err_msg=f"CPMG-{n} FFT != switching function"
            )

    def test_continuous_ramsey_vs_quadrature(self):
        """Continuous Ramsey: FFT matches direct numerical quadrature."""
        omega = 4 * np.pi
        tau = 1.5
        seq = continuous_ramsey_sequence(omega=omega, tau=tau, delta=0.0)
        H_noise = SIGMA_Z / 2

        freqs_fft, F_matrix = fft_filter_function(seq, H_noise, n_samples=4096)
        suscept_fft = noise_susceptibility_from_matrix(F_matrix)

        T = seq.total_duration()
        candidates = np.where(
            (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0) &
            (suscept_fft > 1e-4 * np.max(suscept_fft))
        )[0]
        test_indices = candidates[::len(candidates) // 6][:6]
        test_freqs = freqs_fft[test_indices]

        suscept_quad = _quadrature_susceptibility(seq, H_noise, test_freqs)

        for i, idx in enumerate(test_indices):
            w = freqs_fft[idx]
            rel_err = abs(suscept_fft[idx] - suscept_quad[i]) / max(suscept_quad[i], 1e-15)
            self.assertLess(
                rel_err, 0.02,
                msg=f"ContRamsey at w={w:.2f}: FFT={suscept_fft[idx]:.4e} vs "
                    f"quad={suscept_quad[i]:.4e}, rel_err={rel_err:.4f}"
            )

    def test_continuous_cpmg_vs_quadrature(self):
        """Continuous CPMG: FFT matches direct numerical quadrature."""
        omega = 8 * np.pi
        tau = 1.5
        seq = continuous_cpmg_sequence(omega=omega, tau=tau, n_pulses=2, delta=0.0)
        H_noise = SIGMA_Z / 2

        freqs_fft, F_matrix = fft_filter_function(seq, H_noise, n_samples=4096)
        suscept_fft = noise_susceptibility_from_matrix(F_matrix)

        T = seq.total_duration()
        candidates = np.where(
            (freqs_fft > 2 * np.pi / T) & (freqs_fft < 50.0) &
            (suscept_fft > 1e-4 * np.max(suscept_fft))
        )[0]
        test_indices = candidates[::len(candidates) // 6][:6]
        test_freqs = freqs_fft[test_indices]

        suscept_quad = _quadrature_susceptibility(seq, H_noise, test_freqs)

        for i, idx in enumerate(test_indices):
            w = freqs_fft[idx]
            rel_err = abs(suscept_fft[idx] - suscept_quad[i]) / max(suscept_quad[i], 1e-15)
            self.assertLess(
                rel_err, 0.02,
                msg=f"ContCPMG-2 at w={w:.2f}: FFT={suscept_fft[idx]:.4e} vs "
                    f"quad={suscept_quad[i]:.4e}, rel_err={rel_err:.4f}"
            )

    def test_bloch_components_sum_to_susceptibility(self):
        """Bloch components from FFT matrix sum to total susceptibility."""
        seq = ramsey_sequence(tau=2.0, delta=0.25)
        H_noise = SIGMA_Z / 2
        freqs, F_matrix = fft_filter_function(seq, H_noise, n_samples=4096)

        suscept = noise_susceptibility_from_matrix(F_matrix)
        components = bloch_components_from_matrix(F_matrix)

        component_sum = sum(np.abs(c)**2 for c in components)
        np.testing.assert_allclose(
            component_sum, suscept, rtol=1e-10,
            err_msg="Bloch component sum != total susceptibility"
        )

    def test_ramsey_analytic_formula(self):
        """Ramsey FFT susceptibility matches 4*sin^2(wT/2)/w^2."""
        tau = 2.0
        seq = ramsey_sequence(tau=tau, delta=0.0)
        H_noise = SIGMA_Z / 2
        freqs, F_matrix = fft_filter_function(seq, H_noise, n_samples=8192)

        suscept_fft = noise_susceptibility_from_matrix(F_matrix)
        suscept_formula = 4 * np.sin(freqs * tau / 2)**2 / freqs**2

        significant = suscept_formula > 1e-4 * np.max(suscept_formula)
        mask = (freqs > 1.0) & (freqs < 100.0) & significant
        self.assertTrue(np.any(mask), "No significant frequency points found")
        np.testing.assert_allclose(
            suscept_fft[mask], suscept_formula[mask], rtol=0.02,
            err_msg="Ramsey FFT susceptibility != 4*sin^2(wT/2)/w^2"
        )

    def test_spin_echo_analytic_formula(self):
        """Spin echo FFT susceptibility matches 16*sin^4(wT/4)/w^2."""
        tau = 2.0
        seq = spin_echo_sequence(tau=tau, delta=0.0)
        H_noise = SIGMA_Z / 2
        freqs, F_matrix = fft_filter_function(seq, H_noise, n_samples=8192)

        suscept_fft = noise_susceptibility_from_matrix(F_matrix)
        suscept_formula = 16 * np.sin(freqs * tau / 4)**4 / freqs**2

        significant = suscept_formula > 1e-4 * np.max(suscept_formula)
        mask = (freqs > 1.0) & (freqs < 100.0) & significant
        self.assertTrue(np.any(mask), "No significant frequency points found")
        np.testing.assert_allclose(
            suscept_fft[mask], suscept_formula[mask], rtol=0.02,
            err_msg="SpinEcho FFT susceptibility != 16*sin^4(wT/4)/w^2"
        )


class TestFFTvsAnalytic(unittest.TestCase):
    """Compare FFT filter function against analytic filter function."""

    def _compare_susceptibility(self, seq, label, rtol=0.03):
        """Helper: compare FFT and analytic noise susceptibility."""
        H_noise = SIGMA_Z / 2
        seq.compute_polynomials()
        ff_analytic = seq.get_filter_function_calculator()

        freqs, F_matrix = fft_filter_function(seq, H_noise, n_samples=8192)
        suscept_fft = noise_susceptibility_from_matrix(F_matrix)
        suscept_ana = ff_analytic.noise_susceptibility(freqs)

        T = seq.total_duration()
        mask = (freqs > 2 * np.pi / T) & (freqs < 80.0)
        peak = max(np.max(suscept_fft[mask]), np.max(suscept_ana[mask]))
        sig = (suscept_ana[mask] > 1e-4 * peak) & (suscept_fft[mask] > 1e-4 * peak)
        self.assertTrue(np.any(sig), f"{label}: no significant frequency points")

        rel_err = np.abs(suscept_fft[mask][sig] - suscept_ana[mask][sig]) / suscept_ana[mask][sig]
        self.assertLess(
            np.max(rel_err), rtol,
            msg=f"{label}: max rel error {np.max(rel_err):.4f} exceeds {rtol}"
        )

    def test_ramsey_fft_vs_analytic(self):
        """Ramsey: FFT susceptibility matches analytic."""
        for tau in [1.0, 2.0, 2 * np.pi]:
            seq = ramsey_sequence(tau=tau, delta=0.0)
            self._compare_susceptibility(seq, f"Ramsey(tau={tau:.2f})")

    def test_spin_echo_fft_vs_analytic(self):
        """Spin echo: FFT susceptibility matches analytic."""
        for tau in [1.0, 2.0, 4.0]:
            seq = spin_echo_sequence(tau=tau, delta=0.0)
            self._compare_susceptibility(seq, f"SpinEcho(tau={tau:.2f})")

    def test_cpmg_fft_vs_analytic(self):
        """CPMG-n: FFT susceptibility matches analytic."""
        tau = 2.0
        for n in [1, 2, 4]:
            seq = cpmg_sequence(tau=tau, n_pulses=n, delta=0.0)
            self._compare_susceptibility(seq, f"CPMG-{n}(tau={tau:.2f})")

    def test_continuous_ramsey_fft_vs_analytic(self):
        """Continuous Ramsey: FFT susceptibility matches analytic."""
        for omega in [4 * np.pi, 8 * np.pi]:
            seq = continuous_ramsey_sequence(omega=omega, tau=1.5, delta=0.0)
            self._compare_susceptibility(
                seq, f"ContRamsey(omega={omega:.2f})", rtol=0.05)

    def test_continuous_cpmg_fft_vs_analytic(self):
        """Continuous CPMG: FFT susceptibility matches analytic."""
        omega = 8 * np.pi
        for n in [1, 2]:
            seq = continuous_cpmg_sequence(
                omega=omega, tau=1.5, n_pulses=n, delta=0.0)
            self._compare_susceptibility(
                seq, f"ContCPMG-{n}(omega={omega:.2f})", rtol=0.05)

    def test_bloch_components_fft_vs_analytic(self):
        """Individual Bloch components match between FFT and analytic."""
        seq = spin_echo_sequence(tau=2.0, delta=0.0)
        H_noise = SIGMA_Z / 2
        seq.compute_polynomials()
        ff_analytic = seq.get_filter_function_calculator()

        freqs, F_matrix = fft_filter_function(seq, H_noise, n_samples=8192)
        Fx_fft, Fy_fft, Fz_fft = bloch_components_from_matrix(F_matrix)
        Fx_ana, Fy_ana, Fz_ana = ff_analytic.filter_function(freqs)

        T = seq.total_duration()
        mask = (freqs > 2 * np.pi / T) & (freqs < 80.0)

        # Compare |Fk|^2 for each component
        for name, fft_k, ana_k in [('Fx', Fx_fft, Fx_ana),
                                     ('Fy', Fy_fft, Fy_ana),
                                     ('Fz', Fz_fft, Fz_ana)]:
            s_fft = np.abs(fft_k[mask])**2
            s_ana = np.abs(ana_k[mask])**2
            peak = max(np.max(s_fft), np.max(s_ana))
            if peak < 1e-12:
                continue
            sig = (s_ana > 1e-3 * peak) & (s_fft > 1e-3 * peak)
            if not np.any(sig):
                continue
            rel_err = np.abs(s_fft[sig] - s_ana[sig]) / s_ana[sig]
            self.assertLess(
                np.max(rel_err), 0.03,
                msg=f"SpinEcho {name}: max rel error {np.max(rel_err):.4f}"
            )


if __name__ == '__main__':
    unittest.main()
