"""
Tests for qudit spin-displacement pulse filter functions.

Verifies:
1. Spin-j operators have correct algebraic properties
2. Spin-displacement Hamiltonian reduces to Jx when phases=0
3. D(0, 2*pi) = identity (up to global phase) for all d
4. SU(2) embedding: qudit filter functions = scaling * qubit filter functions
   when driven by pure Jx rotations with Jz noise
5. FFT filter function works for qudit Ramsey, spin echo, CPMG sequences
6. GGM component decomposition sums to total susceptibility
"""

import unittest
import numpy as np
from scipy.linalg import expm

from quantum_pulse_suite.core.spin_displacement import (
    spin_j_operators,
    spin_displacement_hamiltonian,
    spin_displacement_pulse,
    snap_gate,
    su2_scaling_factor,
)
from quantum_pulse_suite.core.qudit_pulse_sequence import (
    QuditPulseSequence,
    SpinDisplacementElement,
    InstantSpinDisplacement,
    QuditFreeEvolution,
)
from quantum_pulse_suite.core.fft_filter_function import (
    fft_filter_function,
    noise_susceptibility_from_matrix,
    bloch_components_from_matrix,
)
from quantum_pulse_suite.core.operators import generalized_gell_mann_matrices
from quantum_pulse_suite.core.irrep_decomposition import (
    angular_momentum_irrep_projectors,
    irrep_resolved_filter_function,
    transition_resolved_filter_function,
    ggm_component_filter_functions,
    irrep_multiplicities,
)
from quantum_pulse_suite import (
    ramsey_sequence,
    spin_echo_sequence,
    cpmg_sequence,
    SIGMA_Z,
)


class TestSpinJOperators(unittest.TestCase):
    """Test angular momentum operator construction."""

    def test_commutation_relations(self):
        """[Jx, Jy] = i*Jz (and cyclic) for d=2,3,4,5."""
        for d in [2, 3, 4, 5]:
            Jx, Jy, Jz = spin_j_operators(d)
            # [Jx, Jy] = i*Jz
            np.testing.assert_allclose(
                Jx @ Jy - Jy @ Jx, 1j * Jz, atol=1e-12,
                err_msg=f"d={d}: [Jx,Jy] != i*Jz"
            )
            # [Jy, Jz] = i*Jx
            np.testing.assert_allclose(
                Jy @ Jz - Jz @ Jy, 1j * Jx, atol=1e-12,
                err_msg=f"d={d}: [Jy,Jz] != i*Jx"
            )
            # [Jz, Jx] = i*Jy
            np.testing.assert_allclose(
                Jz @ Jx - Jx @ Jz, 1j * Jy, atol=1e-12,
                err_msg=f"d={d}: [Jz,Jx] != i*Jy"
            )

    def test_casimir(self):
        """J^2 = j(j+1)*I for d=2,3,4,5."""
        for d in [2, 3, 4, 5]:
            j = (d - 1) / 2
            Jx, Jy, Jz = spin_j_operators(d)
            J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
            np.testing.assert_allclose(
                J2, j * (j + 1) * np.eye(d), atol=1e-12,
                err_msg=f"d={d}: J^2 != j(j+1)*I"
            )

    def test_hermiticity(self):
        """Jx, Jy, Jz are Hermitian."""
        for d in [2, 3, 4, 5]:
            Jx, Jy, Jz = spin_j_operators(d)
            for name, J in [('Jx', Jx), ('Jy', Jy), ('Jz', Jz)]:
                np.testing.assert_allclose(
                    J, J.conj().T, atol=1e-14,
                    err_msg=f"d={d}: {name} not Hermitian"
                )

    def test_jz_eigenvalues(self):
        """Jz eigenvalues are j, j-1, ..., -j."""
        for d in [2, 3, 4, 5]:
            j = (d - 1) / 2
            _, _, Jz = spin_j_operators(d)
            expected = np.array([j - k for k in range(d)])
            np.testing.assert_allclose(
                np.diag(Jz), expected, atol=1e-14,
                err_msg=f"d={d}: Jz diagonal wrong"
            )

    def test_d2_matches_pauli(self):
        """d=2: Jx = sigma_x/2, etc."""
        Jx, Jy, Jz = spin_j_operators(2)
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_allclose(Jx, sx / 2, atol=1e-14)
        np.testing.assert_allclose(Jy, sy / 2, atol=1e-14)
        np.testing.assert_allclose(Jz, sz / 2, atol=1e-14)

    def test_trace_jz_squared(self):
        """Tr(Jz^2) = j(j+1)(2j+1)/3."""
        for d in [2, 3, 4, 5, 6]:
            j = (d - 1) / 2
            _, _, Jz = spin_j_operators(d)
            expected = j * (j + 1) * (2 * j + 1) / 3
            np.testing.assert_allclose(
                np.real(np.trace(Jz @ Jz)), expected, atol=1e-12,
                err_msg=f"d={d}: Tr(Jz^2) wrong"
            )


class TestSpinDisplacementHamiltonian(unittest.TestCase):
    """Test the multi-tone Hamiltonian construction."""

    def test_zero_phases_gives_jx(self):
        """H_rot(phases=0, detunings=0) = omega * Jx."""
        for d in [2, 3, 4, 5]:
            omega = 2.5
            H = spin_displacement_hamiltonian(d, omega=omega)
            Jx, _, _ = spin_j_operators(d)
            np.testing.assert_allclose(
                H, omega * Jx, atol=1e-12,
                err_msg=f"d={d}: H_rot(0) != omega*Jx"
            )

    def test_hermiticity(self):
        """H_rot is Hermitian for arbitrary phases."""
        rng = np.random.default_rng(42)
        for d in [3, 4, 5]:
            phases = rng.uniform(-np.pi, np.pi, d - 1)
            detunings = rng.uniform(-1, 1, d - 1)
            H = spin_displacement_hamiltonian(d, phases, 1.0, detunings)
            np.testing.assert_allclose(
                H, H.conj().T, atol=1e-14,
                err_msg=f"d={d}: H_rot not Hermitian"
            )

    def test_rabi_frequencies(self):
        """Off-diagonal elements match Omega_k/2 = omega*sqrt((k+1)(d-k-1))/2."""
        d = 5
        omega = 3.0
        H = spin_displacement_hamiltonian(d, omega=omega)
        for k in range(d - 1):
            expected_coupling = omega * np.sqrt((k + 1) * (d - k - 1)) / 2.0
            self.assertAlmostEqual(
                abs(H[k, k + 1]), expected_coupling, places=12,
                msg=f"Coupling rate wrong for k={k}"
            )


class TestSpinDisplacementPulse(unittest.TestCase):
    """Test the spin-displacement pulse unitary."""

    def test_unitarity(self):
        """D is unitary for various parameters."""
        rng = np.random.default_rng(123)
        for d in [2, 3, 4, 5]:
            phases = rng.uniform(-np.pi, np.pi, d - 1)
            theta = rng.uniform(0, 2 * np.pi)
            D = spin_displacement_pulse(d, phases, theta)
            np.testing.assert_allclose(
                D @ D.conj().T, np.eye(d), atol=1e-12,
                err_msg=f"d={d}: D not unitary"
            )

    def test_2pi_rotation_identity(self):
        """D(0, 2*pi) should be +/- I (global phase) for integer j.
        For half-integer j, D(0, 4*pi) = I."""
        for d in [2, 3, 4, 5]:
            j = (d - 1) / 2
            # Full period for SU(2): 4*pi for half-integer j, 2*pi for integer j
            period = 4 * np.pi if (2 * j) % 2 == 1 else 2 * np.pi
            D = spin_displacement_pulse(d, theta=period)
            np.testing.assert_allclose(
                D, np.eye(d), atol=1e-10,
                err_msg=f"d={d}: D(0,{period:.1f}) != I"
            )

    def test_d2_matches_qubit_rotation(self):
        """d=2: D(0, theta) = exp(-i*theta*sigma_x/2)."""
        for theta in [np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            D = spin_displacement_pulse(2, theta=theta)
            sx = np.array([[0, 1], [1, 0]], dtype=complex)
            expected = expm(-1j * theta * sx / 2)
            np.testing.assert_allclose(
                D, expected, atol=1e-12,
                err_msg=f"d=2, theta={theta:.2f}: D != exp(-i*theta*sx/2)"
            )


class TestSnapGate(unittest.TestCase):
    """Test virtual SNAP gate."""

    def test_diagonal_unitary(self):
        """SNAP gate is diagonal unitary."""
        phases = [0.1, 0.5, -0.3]
        S = snap_gate(3, phases)
        # Check diagonal
        np.testing.assert_allclose(
            S, np.diag(np.exp(1j * np.array(phases))), atol=1e-14
        )
        # Check unitary
        np.testing.assert_allclose(
            S @ S.conj().T, np.eye(3), atol=1e-14
        )


class TestQuditPulseSequence(unittest.TestCase):
    """Test QuditPulseSequence construction and properties."""

    def test_total_duration(self):
        """Total duration sums element durations."""
        seq = QuditPulseSequence(3)
        seq.add_spin_displacement(theta=np.pi, omega=10.0)
        seq.add_free_evolution(tau=1.0)
        seq.add_spin_displacement(theta=np.pi, omega=10.0)
        expected = np.pi / 10 + 1.0 + np.pi / 10
        self.assertAlmostEqual(seq.total_duration(), expected, places=10)

    def test_instant_has_zero_duration(self):
        """Instantaneous pulses contribute zero duration."""
        seq = QuditPulseSequence(3)
        seq.add_instant_spin_displacement(theta=np.pi)
        seq.add_free_evolution(tau=2.0)
        seq.add_instant_spin_displacement(theta=np.pi)
        self.assertAlmostEqual(seq.total_duration(), 2.0, places=12)

    def test_ramsey_factory(self):
        """Ramsey factory creates correct structure."""
        for d in [2, 3, 4]:
            seq = QuditPulseSequence.ramsey(d, tau=1.0, continuous=False)
            self.assertEqual(len(seq.elements), 3)
            self.assertAlmostEqual(seq.total_duration(), 1.0)

    def test_total_unitary_is_unitary(self):
        """Total unitary is actually unitary."""
        rng = np.random.default_rng(99)
        for d in [3, 4]:
            seq = QuditPulseSequence(d)
            for _ in range(3):
                phases = rng.uniform(-np.pi, np.pi, d - 1)
                theta = rng.uniform(0.5, 2.0)
                seq.add_instant_spin_displacement(phases, theta)
            U = seq.total_unitary()
            np.testing.assert_allclose(
                U @ U.conj().T, np.eye(d), atol=1e-12,
                err_msg=f"d={d}: total unitary not unitary"
            )


class TestSU2EmbeddingFilterFunction(unittest.TestCase):
    """
    Core test: qudit filter functions under SU(2) rotations with Jz noise
    should equal the qubit filter function scaled by d(d^2-1)/6.
    """

    def _compare_qudit_to_qubit(self, qubit_seq, qudit_seq, d, label, rtol=0.03):
        """Compare qudit FFT susceptibility to scaled qubit susceptibility."""
        # Qubit: noise = sigma_z / 2
        H_noise_qubit = SIGMA_Z / 2
        freqs_q, F_mat_q = fft_filter_function(qubit_seq, H_noise_qubit, n_samples=8192)
        suscept_qubit = noise_susceptibility_from_matrix(F_mat_q)

        # Qudit: noise = Jz
        _, _, Jz = spin_j_operators(d)
        freqs_d, F_mat_d = fft_filter_function(qudit_seq, Jz, n_samples=8192)
        suscept_qudit = noise_susceptibility_from_matrix(F_mat_d)

        scaling = su2_scaling_factor(d)

        # Compare at significant frequencies
        T = qubit_seq.total_duration()
        mask = (freqs_q > 2 * np.pi / T) & (freqs_q < 50.0)
        peak = max(np.max(suscept_qubit[mask]), 1e-15)
        sig = suscept_qubit[mask] > 1e-4 * peak

        if not np.any(sig):
            self.fail(f"{label}: no significant frequency points")

        # The qudit susceptibility should be scaling * qubit susceptibility
        ratio = suscept_qudit[mask][sig] / suscept_qubit[mask][sig]
        np.testing.assert_allclose(
            ratio, scaling, rtol=rtol,
            err_msg=f"{label} d={d}: ratio should be {scaling:.1f}, "
                    f"got {np.mean(ratio):.4f}"
        )

    def test_ramsey_instantaneous(self):
        """Instantaneous Ramsey: qudit/qubit scaling for d=3,4,5."""
        tau = 2.0
        qubit_seq = ramsey_sequence(tau=tau, delta=0.0)
        for d in [3, 4, 5]:
            qudit_seq = QuditPulseSequence.ramsey(d, tau=tau, continuous=False)
            self._compare_qudit_to_qubit(
                qubit_seq, qudit_seq, d,
                f"InstRamsey d={d}"
            )

    def test_spin_echo_instantaneous(self):
        """Instantaneous spin echo: qudit/qubit scaling for d=3,4."""
        tau = 2.0
        qubit_seq = spin_echo_sequence(tau=tau, delta=0.0)
        for d in [3, 4]:
            qudit_seq = QuditPulseSequence.spin_echo(d, tau=tau, continuous=False)
            self._compare_qudit_to_qubit(
                qubit_seq, qudit_seq, d,
                f"InstSpinEcho d={d}"
            )

    def test_cpmg_instantaneous(self):
        """Instantaneous CPMG: qudit/qubit scaling for d=3."""
        tau = 2.0
        n_pulses = 2
        qubit_seq = cpmg_sequence(tau=tau, n_pulses=n_pulses, delta=0.0)
        d = 3
        qudit_seq = QuditPulseSequence.cpmg(d, tau=tau, n_pulses=n_pulses,
                                             continuous=False)
        self._compare_qudit_to_qubit(
            qubit_seq, qudit_seq, d,
            f"InstCPMG-{n_pulses} d={d}"
        )

    def test_scaling_factor_values(self):
        """Verify scaling factor d(d^2-1)/6."""
        self.assertAlmostEqual(su2_scaling_factor(2), 1.0)
        self.assertAlmostEqual(su2_scaling_factor(3), 4.0)
        self.assertAlmostEqual(su2_scaling_factor(4), 10.0)
        self.assertAlmostEqual(su2_scaling_factor(5), 20.0)


class TestQuditFFTFilterFunction(unittest.TestCase):
    """Test that FFT filter function works correctly for qudit sequences."""

    def test_ramsey_analytic_formula(self):
        """Qudit Ramsey (instantaneous): susceptibility matches
        scaling * 4*sin^2(wT/2)/w^2."""
        for d in [2, 3, 4]:
            tau = 2.0
            seq = QuditPulseSequence.ramsey(d, tau=tau, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=8192)
            suscept = noise_susceptibility_from_matrix(F_mat)

            scaling = su2_scaling_factor(d)
            expected = scaling * 4 * np.sin(freqs * tau / 2)**2 / freqs**2

            mask = (freqs > 1.0) & (freqs < 100.0)
            sig = expected[mask] > 1e-4 * np.max(expected[mask])
            self.assertTrue(np.any(sig), f"d={d}: no significant points")
            np.testing.assert_allclose(
                suscept[mask][sig], expected[mask][sig], rtol=0.02,
                err_msg=f"d={d}: Ramsey FFT != scaling*4sin^2(wT/2)/w^2"
            )

    def test_continuous_ramsey_runs(self):
        """Continuous qudit Ramsey computes without error."""
        for d in [3, 4]:
            omega = 4 * np.pi
            seq = QuditPulseSequence.ramsey(d, tau=1.0, omega=omega,
                                            continuous=True)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=2048)
            suscept = noise_susceptibility_from_matrix(F_mat)
            self.assertTrue(np.all(np.isfinite(suscept)))
            self.assertGreater(np.max(suscept), 0)

    def test_spin_echo_analytic_formula(self):
        """Qudit spin echo (instantaneous): susceptibility matches
        scaling * 16*sin^4(wT/4)/w^2."""
        for d in [2, 3]:
            tau = 2.0
            seq = QuditPulseSequence.spin_echo(d, tau=tau, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=8192)
            suscept = noise_susceptibility_from_matrix(F_mat)

            scaling = su2_scaling_factor(d)
            expected = scaling * 16 * np.sin(freqs * tau / 4)**4 / freqs**2

            mask = (freqs > 1.0) & (freqs < 100.0)
            sig = expected[mask] > 1e-4 * np.max(expected[mask])
            self.assertTrue(np.any(sig), f"d={d}: no significant points")
            np.testing.assert_allclose(
                suscept[mask][sig], expected[mask][sig], rtol=0.02,
                err_msg=f"d={d}: SpinEcho FFT != scaling*16sin^4(wT/4)/w^2"
            )


class TestGGMDecomposition(unittest.TestCase):
    """Test GGM component decomposition for qudit filter functions."""

    def test_components_sum_to_susceptibility(self):
        """Sum of |R_a|^2 over GGM components = noise susceptibility."""
        for d in [3, 4]:
            seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

            suscept = noise_susceptibility_from_matrix(F_mat)
            ggm = generalized_gell_mann_matrices(d)
            components = bloch_components_from_matrix(F_mat, ggm)

            component_sum = sum(np.abs(c)**2 for c in components)

            mask = freqs > 1.0
            np.testing.assert_allclose(
                component_sum[mask], suscept[mask], rtol=1e-8,
                err_msg=f"d={d}: GGM component sum != susceptibility"
            )

    def test_su2_only_j_components(self):
        """Under pure SU(2) rotations with Jz noise, only the 3 angular
        momentum components (those corresponding to Jx, Jy, Jz in GGM basis)
        should be nonzero."""
        d = 3
        seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
        _, _, Jz = spin_j_operators(d)
        freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

        ggm = generalized_gell_mann_matrices(d)
        components = bloch_components_from_matrix(F_mat, ggm)

        # For d=3, the GGM matrices are lambda_1..lambda_8 (Gell-Mann)
        # Under SU(2) (Jx,Jy,Jz), only the spin-1 triplet should contribute.
        # These correspond to lambda_1 (Jx-like: 01+10), lambda_2 (Jy-like),
        # lambda_3 (Jz-like: diag(1,-1,0))...
        # Actually for spin-1, Jx,Jy,Jz map to specific linear combinations
        # of Gell-Mann matrices. The key point is that only 3 out of 8
        # components should be nonzero.

        # Compute total power in each component
        powers = [np.sum(np.abs(c)**2) for c in components]
        total_power = sum(powers)

        # The 3 Jx/Jy/Jz directions should capture all the power
        # Sort and check top 3 have ~100% of power
        sorted_powers = sorted(powers, reverse=True)
        top3_fraction = sum(sorted_powers[:3]) / total_power

        self.assertGreater(
            top3_fraction, 0.999,
            msg=f"d={d}: top 3 GGM components have {top3_fraction:.6f} of power, "
                f"expected ~1.0 for SU(2) rotation"
        )


class TestFromPulseParams(unittest.TestCase):
    """Test building sequences from parameter lists (MQS Prog interface)."""

    def test_from_pulse_params_unitary(self):
        """from_pulse_params produces valid sequence with unitary total."""
        d = 3
        rng = np.random.default_rng(42)
        n_pulses = d
        phases_list = [rng.uniform(-np.pi, np.pi, d - 1) for _ in range(n_pulses)]
        thetas = rng.uniform(0.5, 2.0, n_pulses).tolist()

        seq = QuditPulseSequence.from_pulse_params(
            d, phases_list, thetas, omega=5.0, continuous=False
        )
        U = seq.total_unitary()
        np.testing.assert_allclose(
            U @ U.conj().T, np.eye(d), atol=1e-12,
            err_msg="from_pulse_params: total unitary not unitary"
        )

    def test_from_pulse_params_fft(self):
        """from_pulse_params sequence works with fft_filter_function."""
        d = 3
        rng = np.random.default_rng(42)
        n_pulses = d
        phases_list = [rng.uniform(-np.pi, np.pi, d - 1) for _ in range(n_pulses)]
        thetas = rng.uniform(0.5, 2.0, n_pulses).tolist()

        seq = QuditPulseSequence.from_pulse_params(
            d, phases_list, thetas, omega=5.0, continuous=True
        )
        _, _, Jz = spin_j_operators(d)
        freqs, F_mat = fft_filter_function(seq, Jz, n_samples=2048)
        suscept = noise_susceptibility_from_matrix(F_mat)

        self.assertTrue(np.all(np.isfinite(suscept)))
        self.assertGreater(np.max(suscept), 0)


class TestIrrepProjectors(unittest.TestCase):
    """Test angular momentum irrep projectors in Liouville space."""

    def test_projectors_are_projectors(self):
        """P_L^2 = P_L and P_L = P_L^dag for all L."""
        for d in [2, 3, 4]:
            projectors = angular_momentum_irrep_projectors(d)
            for L, P in projectors.items():
                # Idempotent
                np.testing.assert_allclose(
                    P @ P, P, atol=1e-10,
                    err_msg=f"d={d}, L={L}: P_L^2 != P_L"
                )
                # Hermitian
                np.testing.assert_allclose(
                    P, P.conj().T, atol=1e-10,
                    err_msg=f"d={d}, L={L}: P_L not Hermitian"
                )

    def test_projectors_are_orthogonal(self):
        """P_L @ P_{L'} = 0 for L != L'."""
        for d in [2, 3, 4]:
            projectors = angular_momentum_irrep_projectors(d)
            L_values = sorted(projectors.keys())
            for i, L1 in enumerate(L_values):
                for L2 in L_values[i+1:]:
                    product = projectors[L1] @ projectors[L2]
                    np.testing.assert_allclose(
                        product, 0, atol=1e-10,
                        err_msg=f"d={d}: P_{L1} @ P_{L2} != 0"
                    )

    def test_projectors_sum_to_identity(self):
        """Sum of all P_L = I_{d^2}."""
        for d in [2, 3, 4, 5]:
            projectors = angular_momentum_irrep_projectors(d)
            total = sum(projectors.values())
            np.testing.assert_allclose(
                total, np.eye(d * d), atol=1e-10,
                err_msg=f"d={d}: sum of P_L != I"
            )

    def test_correct_multiplicities(self):
        """Each irrep L has dimension 2L+1."""
        for d in [2, 3, 4, 5]:
            projectors = angular_momentum_irrep_projectors(d)
            expected = irrep_multiplicities(d)
            for L, P in projectors.items():
                rank = int(round(np.real(np.trace(P))))
                self.assertEqual(
                    rank, expected[L],
                    msg=f"d={d}, L={L}: rank={rank}, expected {expected[L]}"
                )

    def test_l0_is_identity_component(self):
        """L=0 projector picks out the trace (identity) component."""
        for d in [2, 3, 4]:
            projectors = angular_momentum_irrep_projectors(d)
            P0 = projectors[0]
            # The identity operator vectorized
            Id_vec = np.eye(d, dtype=complex).reshape(d * d)
            # P_0 should project onto span of Id_vec (normalized)
            Id_norm = Id_vec / np.linalg.norm(Id_vec)
            projected = P0 @ Id_norm
            # Should recover Id_norm (up to phase)
            overlap = np.abs(np.dot(projected.conj(), Id_norm))
            np.testing.assert_allclose(
                overlap, 1.0, atol=1e-10,
                err_msg=f"d={d}: P_0 doesn't project onto identity"
            )

    def test_jz_in_l1_sector(self):
        """Jz should live entirely in the L=1 sector."""
        for d in [2, 3, 4, 5]:
            _, _, Jz = spin_j_operators(d)
            Jz_vec = Jz.reshape(d * d)
            projectors = angular_momentum_irrep_projectors(d)

            # Project Jz onto each sector
            for L, P in projectors.items():
                proj = P @ Jz_vec
                norm_sq = np.real(np.dot(proj.conj(), proj))
                if L == 1:
                    # Jz should have nonzero projection onto L=1
                    self.assertGreater(
                        norm_sq, 0.1,
                        msg=f"d={d}: Jz has no L=1 component"
                    )
                else:
                    # Jz should have zero projection onto other sectors
                    np.testing.assert_allclose(
                        norm_sq, 0, atol=1e-10,
                        err_msg=f"d={d}: Jz has nonzero L={L} component"
                    )


class TestIrrepResolvedFilter(unittest.TestCase):
    """Test irrep-resolved filter function decomposition."""

    def test_irrep_components_sum_to_total(self):
        """Sum over L of F_L(w) = total susceptibility."""
        for d in [3, 4, 5]:
            seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

            total = noise_susceptibility_from_matrix(F_mat)
            irrep_susc = irrep_resolved_filter_function(F_mat, d)
            irrep_sum = sum(irrep_susc.values())

            mask = freqs > 1.0
            np.testing.assert_allclose(
                irrep_sum[mask], total[mask], rtol=1e-8,
                err_msg=f"d={d}: irrep sum != total susceptibility"
            )

    def test_jz_noise_only_l1(self):
        """For Jz dephasing noise with SU(2) pulses, only L=1 contributes.

        Since Jz is in the L=1 sector and the SU(2) rotations preserve
        angular momentum sectors, the toggling-frame Hamiltonian h(t) = U†JzU
        stays entirely in L=1.
        """
        for d in [3, 4]:
            seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

            irrep_susc = irrep_resolved_filter_function(F_mat, d)

            # Total power
            total = noise_susceptibility_from_matrix(F_mat)
            mask = freqs > 1.0
            total_power = np.sum(total[mask])

            for L, F_L in irrep_susc.items():
                power_L = np.sum(F_L[mask])
                if L == 1:
                    # L=1 should have ~all the power
                    np.testing.assert_allclose(
                        power_L, total_power, rtol=1e-6,
                        err_msg=f"d={d}: L=1 doesn't capture all power"
                    )
                else:
                    # Other sectors should have negligible power
                    self.assertLess(
                        power_L / total_power, 1e-8,
                        msg=f"d={d}: L={L} has {power_L/total_power:.2e} "
                            f"fraction of total power"
                    )

    def test_spin_echo_jz_only_l1(self):
        """Spin echo with SU(2) pulses and Jz noise: only L=1."""
        d = 3
        seq = QuditPulseSequence.spin_echo(d, tau=2.0, continuous=False)
        _, _, Jz = spin_j_operators(d)
        freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

        irrep_susc = irrep_resolved_filter_function(F_mat, d)
        total = noise_susceptibility_from_matrix(F_mat)
        mask = freqs > 1.0
        total_power = np.sum(total[mask])

        l1_power = np.sum(irrep_susc[1][mask])
        np.testing.assert_allclose(
            l1_power, total_power, rtol=1e-6,
            err_msg=f"d={d}: spin echo L=1 doesn't capture all power"
        )

    def test_qubit_limit_single_irrep(self):
        """For d=2, there are L=0 and L=1 sectors only.
        With Jz noise, all power in L=1 (3-dimensional = Bloch vector)."""
        d = 2
        seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
        _, _, Jz = spin_j_operators(d)
        freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

        irrep_susc = irrep_resolved_filter_function(F_mat, d)

        # Should have L=0 and L=1
        self.assertIn(0, irrep_susc)
        self.assertIn(1, irrep_susc)

        mask = freqs > 1.0
        # L=0 should be zero (Jz is traceless, stays traceless under rotation)
        np.testing.assert_allclose(
            irrep_susc[0][mask], 0, atol=1e-12,
            err_msg="d=2: L=0 should be zero for traceless noise"
        )
        # L=1 should equal total
        total = noise_susceptibility_from_matrix(F_mat)
        np.testing.assert_allclose(
            irrep_susc[1][mask], total[mask], rtol=1e-8,
            err_msg="d=2: L=1 should equal total for Jz noise"
        )


class TestTransitionResolved(unittest.TestCase):
    """Test transition-resolved filter function decomposition."""

    def test_transition_components_sum_to_total(self):
        """Sum of all transition components = total susceptibility."""
        for d in [3, 4]:
            seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

            total = noise_susceptibility_from_matrix(F_mat)
            trans_susc = transition_resolved_filter_function(F_mat, d)
            trans_sum = sum(trans_susc.values())

            mask = freqs > 1.0
            np.testing.assert_allclose(
                trans_sum[mask], total[mask], rtol=1e-8,
                err_msg=f"d={d}: transition sum != total"
            )

    def test_d2_matches_qubit_bloch(self):
        """For d=2, transition (0,1) gives the off-diagonal (Fx+Fy)
        and diag gives Fz, which should sum to total."""
        d = 2
        seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
        _, _, Jz = spin_j_operators(d)
        freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

        trans_susc = transition_resolved_filter_function(F_mat, d)
        total = noise_susceptibility_from_matrix(F_mat)

        combined = trans_susc[('diag',)] + trans_susc[(0, 1)]
        mask = freqs > 1.0
        np.testing.assert_allclose(
            combined[mask], total[mask], rtol=1e-8,
            err_msg="d=2: diag + (0,1) != total"
        )


class TestGGMComponentFilter(unittest.TestCase):
    """Test GGM component filter function decomposition."""

    def test_ggm_components_sum_to_susceptibility(self):
        """Sum of |R_a|^2 = total susceptibility."""
        for d in [3, 4]:
            seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=4096)

            total = noise_susceptibility_from_matrix(F_mat)
            susc_components, labels = ggm_component_filter_functions(F_mat, d)

            component_sum = np.sum(susc_components, axis=0)

            mask = freqs > 1.0
            np.testing.assert_allclose(
                component_sum[mask], total[mask], rtol=1e-8,
                err_msg=f"d={d}: GGM component sum != susceptibility"
            )

    def test_correct_number_of_components(self):
        """Should have d^2-1 components."""
        for d in [3, 4, 5]:
            seq = QuditPulseSequence.ramsey(d, tau=2.0, continuous=False)
            _, _, Jz = spin_j_operators(d)
            freqs, F_mat = fft_filter_function(seq, Jz, n_samples=1024)
            susc, labels = ggm_component_filter_functions(F_mat, d)
            self.assertEqual(susc.shape[0], d**2 - 1, f"d={d}: wrong count")
            self.assertEqual(len(labels), d**2 - 1, f"d={d}: wrong label count")

    def test_labels_format(self):
        """Labels should follow the sym/asym/diag naming convention."""
        d = 3
        seq = QuditPulseSequence.ramsey(d, tau=1.0, continuous=False)
        _, _, Jz = spin_j_operators(d)
        freqs, F_mat = fft_filter_function(seq, Jz, n_samples=1024)
        _, labels = ggm_component_filter_functions(F_mat, d)

        # d=3: 3 sym, 3 asym, 2 diag = 8 total
        self.assertEqual(len(labels), 8)
        sym_labels = [l for l in labels if l.startswith('sym')]
        asym_labels = [l for l in labels if l.startswith('asym')]
        diag_labels = [l for l in labels if l.startswith('diag')]
        self.assertEqual(len(sym_labels), 3)
        self.assertEqual(len(asym_labels), 3)
        self.assertEqual(len(diag_labels), 2)


if __name__ == '__main__':
    unittest.main()
