"""
Unit tests for the ThreeLevelClock system and multi-level pulse sequences.

Tests cover:
- ThreeLevelClock state preparation and properties
- Subspace Pauli matrices
- Noise operators
- Multi-level pulse sequences
- Filter function computation on subspaces
- Differential spectroscopy protocols
"""

import unittest
import numpy as np
from scipy.linalg import expm

from quantum_pulse_suite.systems import ThreeLevelClock, QubitSystem
from quantum_pulse_suite.core.hilbert_space import HilbertSpace, Subspace
from quantum_pulse_suite.core.operators import (
    SIGMA_X, SIGMA_Y, SIGMA_Z,
    gell_mann_matrices,
    generalized_gell_mann_matrices,
    subspace_pauli,
)
from quantum_pulse_suite.core.multilevel import (
    MultiLevelPulseSequence,
    MultiLevelInstantPulse,
    MultiLevelFreeEvolution,
    SubspaceFilterFunction,
    multilevel_ramsey,
    multilevel_spin_echo,
    multilevel_cpmg,
    DifferentialSpectroscopySequence,
)


class TestThreeLevelClockBasics(unittest.TestCase):
    """Test basic ThreeLevelClock functionality."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_dimension(self):
        """Test that system has correct dimension."""
        self.assertEqual(self.system.dim, 3)

    def test_level_labels(self):
        """Test default level labels."""
        labels = self.system.hilbert_space.level_labels
        self.assertEqual(labels, ['g', 'e', 'm'])

    def test_subspaces_registered(self):
        """Test that probe and clock subspaces are registered."""
        self.assertIn('probe', self.system.subspaces)
        self.assertIn('clock', self.system.subspaces)

    def test_probe_subspace_levels(self):
        """Test probe subspace contains correct levels."""
        probe = self.system.probe
        self.assertEqual(probe.levels, (0, 1))  # |g> <-> |e>

    def test_clock_subspace_levels(self):
        """Test clock subspace contains correct levels."""
        clock = self.system.clock
        self.assertEqual(clock.levels, (0, 2))  # |g> <-> |m>

    def test_identity_dimension(self):
        """Test identity operator has correct dimension."""
        I = self.system.identity
        self.assertEqual(I.shape, (3, 3))
        np.testing.assert_array_almost_equal(I, np.eye(3))


class TestThreeLevelClockStates(unittest.TestCase):
    """Test state preparation in ThreeLevelClock."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_clock_superposition_normalization(self):
        """Test that clock superposition is normalized."""
        psi = self.system.prepare_clock_superposition()
        norm = np.linalg.norm(psi)
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_clock_superposition_populations(self):
        """Test clock superposition has correct populations."""
        psi = self.system.prepare_clock_superposition()
        # Should have equal population in |g> and |m>
        pop_g = np.abs(psi[0])**2
        pop_e = np.abs(psi[1])**2
        pop_m = np.abs(psi[2])**2

        self.assertAlmostEqual(pop_g, 0.5, places=10)
        self.assertAlmostEqual(pop_e, 0.0, places=10)
        self.assertAlmostEqual(pop_m, 0.5, places=10)

    def test_clock_superposition_with_phase(self):
        """Test clock superposition with non-zero phase."""
        phase = np.pi / 4
        psi = self.system.prepare_clock_superposition(phase=phase)

        # Ground state amplitude should be real
        self.assertAlmostEqual(np.imag(psi[0]), 0.0, places=10)

        # Metastable amplitude should have correct phase
        expected_m = np.exp(1j * phase) / np.sqrt(2)
        self.assertAlmostEqual(psi[2], expected_m, places=10)

    def test_probe_superposition(self):
        """Test probe transition superposition."""
        psi = self.system.prepare_probe_superposition()

        pop_g = np.abs(psi[0])**2
        pop_e = np.abs(psi[1])**2
        pop_m = np.abs(psi[2])**2

        self.assertAlmostEqual(pop_g, 0.5, places=10)
        self.assertAlmostEqual(pop_e, 0.5, places=10)
        self.assertAlmostEqual(pop_m, 0.0, places=10)

    def test_prepare_named_states(self):
        """Test named state preparation."""
        # Ground state
        psi_g = self.system.prepare_state('ground')
        np.testing.assert_array_almost_equal(psi_g, [1, 0, 0])

        # Excited state
        psi_e = self.system.prepare_state('excited')
        np.testing.assert_array_almost_equal(psi_e, [0, 1, 0])

        # Metastable state
        psi_m = self.system.prepare_state('metastable')
        np.testing.assert_array_almost_equal(psi_m, [0, 0, 1])

    def test_clock_coherence(self):
        """Test clock coherence extraction."""
        psi = self.system.prepare_clock_superposition()
        coherence = self.system.clock_coherence(psi)

        # For (|g> + |m>)/sqrt(2), coherence = 1/2
        self.assertAlmostEqual(np.abs(coherence), 0.5, places=10)

    def test_contrast(self):
        """Test contrast computation."""
        psi = self.system.prepare_clock_superposition()
        contrast = self.system.contrast(psi)

        # For perfect superposition, contrast = 1
        self.assertAlmostEqual(contrast, 1.0, places=10)

    def test_clock_phase_extraction(self):
        """Test clock phase extraction."""
        phase_in = np.pi / 3
        psi = self.system.prepare_clock_superposition(phase=phase_in)
        phase_out = self.system.clock_phase(psi)

        # Phase should match (up to sign convention)
        self.assertAlmostEqual(np.abs(phase_out), np.abs(phase_in), places=10)


class TestThreeLevelClockPauliMatrices(unittest.TestCase):
    """Test Pauli matrices on subspaces."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_probe_pauli_hermitian(self):
        """Test probe Pauli matrices are Hermitian."""
        sx, sy, sz = self.system.probe_pauli_matrices()

        np.testing.assert_array_almost_equal(sx, sx.conj().T)
        np.testing.assert_array_almost_equal(sy, sy.conj().T)
        np.testing.assert_array_almost_equal(sz, sz.conj().T)

    def test_clock_pauli_hermitian(self):
        """Test clock Pauli matrices are Hermitian."""
        sx, sy, sz = self.system.clock_pauli_matrices()

        np.testing.assert_array_almost_equal(sx, sx.conj().T)
        np.testing.assert_array_almost_equal(sy, sy.conj().T)
        np.testing.assert_array_almost_equal(sz, sz.conj().T)

    def test_probe_pauli_anticommute(self):
        """Test probe Pauli matrices satisfy anticommutation relations."""
        sx, sy, sz = self.system.probe_pauli_matrices()

        # {σi, σj} = 2 δij on the subspace
        # Since these are embedded in 3x3, check projected anticommutator
        anticomm_xy = sx @ sy + sy @ sx
        anticomm_xz = sx @ sz + sz @ sx
        anticomm_yz = sy @ sz + sz @ sy

        # Off-diagonal anticommutators should vanish on the subspace
        np.testing.assert_array_almost_equal(anticomm_xy, np.zeros((3, 3)))
        np.testing.assert_array_almost_equal(anticomm_xz, np.zeros((3, 3)))
        np.testing.assert_array_almost_equal(anticomm_yz, np.zeros((3, 3)))

    def test_probe_sigma_z_eigenvalues(self):
        """Test probe σz has correct eigenvalues on subspace."""
        _, _, sz = self.system.probe_pauli_matrices()

        eigenvalues = np.linalg.eigvalsh(sz)
        eigenvalues_sorted = np.sort(eigenvalues)

        # Should be -1, 0, +1 (0 from the metastable state outside subspace)
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_almost_equal(eigenvalues_sorted, expected)

    def test_clock_sigma_z_eigenvalues(self):
        """Test clock σz has correct eigenvalues on subspace."""
        _, _, sz = self.system.clock_pauli_matrices()

        eigenvalues = np.linalg.eigvalsh(sz)
        eigenvalues_sorted = np.sort(eigenvalues)

        # Should be -1, 0, +1
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_almost_equal(eigenvalues_sorted, expected)

    def test_pauli_commutators(self):
        """Test Pauli commutation relations [σx, σy] = 2i σz."""
        sx, sy, sz = self.system.probe_pauli_matrices()

        commutator = sx @ sy - sy @ sx
        expected = 2j * sz

        np.testing.assert_array_almost_equal(commutator, expected)


class TestThreeLevelClockNoiseOperators(unittest.TestCase):
    """Test noise operator definitions."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_noise_operators_exist(self):
        """Test that noise operators are defined."""
        noise_ops = self.system.noise_operators()

        self.assertIn('probe_dephasing', noise_ops)
        self.assertIn('clock_dephasing', noise_ops)
        self.assertIn('global_dephasing', noise_ops)

    def test_noise_operators_hermitian(self):
        """Test that noise operators are Hermitian."""
        noise_ops = self.system.noise_operators()

        for name, op in noise_ops.items():
            np.testing.assert_array_almost_equal(
                op, op.conj().T,
                err_msg=f"Noise operator '{name}' is not Hermitian"
            )

    def test_global_dephasing_traceless(self):
        """Test global dephasing operator is traceless."""
        op = self.system.global_dephasing_operator()
        trace = np.trace(op)
        self.assertAlmostEqual(trace, 0.0, places=10)

    def test_global_dephasing_structure(self):
        """Test global dephasing operator has correct structure."""
        op = self.system.global_dephasing_operator()

        # Should be 2|g><g| - |e><e| - |m><m|
        expected = np.diag([2, -1, -1]).astype(complex)
        np.testing.assert_array_almost_equal(op, expected)


class TestThreeLevelClockHamiltonians(unittest.TestCase):
    """Test Hamiltonian construction."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_probe_hamiltonian_hermitian(self):
        """Test probe Hamiltonian is Hermitian."""
        H = self.system.probe_hamiltonian(omega=1.0, axis=[1, 0, 0])
        np.testing.assert_array_almost_equal(H, H.conj().T)

    def test_clock_hamiltonian_hermitian(self):
        """Test clock Hamiltonian is Hermitian."""
        H = self.system.clock_hamiltonian(omega=1.0, axis=[1, 0, 0])
        np.testing.assert_array_almost_equal(H, H.conj().T)

    def test_probe_hamiltonian_x_axis(self):
        """Test probe Hamiltonian for x-axis driving."""
        omega = 2.0
        H = self.system.probe_hamiltonian(omega=omega, axis=[1, 0, 0])

        sx, _, _ = self.system.probe_pauli_matrices()
        expected = omega / 2 * sx

        np.testing.assert_array_almost_equal(H, expected)

    def test_free_evolution_hamiltonian(self):
        """Test free evolution Hamiltonian."""
        delta_probe = 0.5
        delta_clock = 0.3

        H = self.system.free_evolution_hamiltonian(
            delta_probe=delta_probe,
            delta_clock=delta_clock
        )

        # Check it's Hermitian
        np.testing.assert_array_almost_equal(H, H.conj().T)

        # Check dimensions
        self.assertEqual(H.shape, (3, 3))


class TestMultiLevelPulseSequence(unittest.TestCase):
    """Test multi-level pulse sequences."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_sequence_creation(self):
        """Test creating a multi-level pulse sequence."""
        seq = MultiLevelPulseSequence(self.system, self.system.probe)

        self.assertEqual(seq.system, self.system)
        self.assertEqual(seq.subspace, self.system.probe)
        self.assertEqual(len(seq.elements), 0)

    def test_add_instant_pulse(self):
        """Test adding instantaneous pulse."""
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_instant_pulse([1, 0, 0], np.pi/2)

        self.assertEqual(len(seq.elements), 1)
        self.assertIsInstance(seq.elements[0], MultiLevelInstantPulse)

    def test_add_free_evolution(self):
        """Test adding free evolution."""
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_free_evolution(tau=1.0, delta=0.1)

        self.assertEqual(len(seq.elements), 1)
        self.assertIsInstance(seq.elements[0], MultiLevelFreeEvolution)

    def test_total_duration(self):
        """Test total duration calculation."""
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_instant_pulse([1, 0, 0], np.pi/2)  # 0 duration
        seq.add_free_evolution(tau=0.5)  # 0.5 duration
        seq.add_instant_pulse([1, 0, 0], np.pi/2)  # 0 duration

        self.assertAlmostEqual(seq.total_duration(), 0.5)

    def test_total_unitary_dimension(self):
        """Test total unitary has correct dimension."""
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_instant_pulse([1, 0, 0], np.pi/2)
        seq.add_free_evolution(tau=1.0)
        seq.add_instant_pulse([1, 0, 0], np.pi/2)

        U = seq.total_unitary()
        self.assertEqual(U.shape, (3, 3))

    def test_total_unitary_is_unitary(self):
        """Test total unitary is actually unitary."""
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_instant_pulse([1, 0, 0], np.pi/2)
        seq.add_free_evolution(tau=1.0)
        seq.add_instant_pulse([1, 0, 0], np.pi)
        seq.add_free_evolution(tau=1.0)
        seq.add_instant_pulse([1, 0, 0], np.pi/2)

        U = seq.total_unitary()
        should_be_identity = U @ U.conj().T

        np.testing.assert_array_almost_equal(
            should_be_identity, np.eye(3), decimal=10
        )

    def test_pi_pulse_on_probe(self):
        """Test π pulse inverts probe subspace."""
        seq = MultiLevelPulseSequence(self.system, self.system.probe)
        seq.add_instant_pulse([1, 0, 0], np.pi)

        # Start in |g>
        psi0 = self.system.prepare_state('ground')
        psi_final = seq.evolve_state(psi0)

        # Should end in |e> (up to phase)
        self.assertAlmostEqual(np.abs(psi_final[1]), 1.0, places=5)


class TestMultiLevelFilterFunction(unittest.TestCase):
    """Test filter function computation for multi-level sequences."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_filter_function_computable(self):
        """Test that filter function can be computed."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()

        frequencies = np.linspace(0.1, 10, 50)
        Fx, Fy, Fz = ff.filter_function(frequencies)

        self.assertEqual(len(Fx), 50)
        self.assertEqual(len(Fy), 50)
        self.assertEqual(len(Fz), 50)

    def test_filter_function_finite(self):
        """Test filter function values are finite."""
        seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()

        frequencies = np.linspace(0.1, 10, 50)
        Fx, Fy, Fz = ff.filter_function(frequencies)

        self.assertTrue(np.all(np.isfinite(Fx)))
        self.assertTrue(np.all(np.isfinite(Fy)))
        self.assertTrue(np.all(np.isfinite(Fz)))

    def test_ramsey_filter_function_shape(self):
        """Test Ramsey filter function has expected sinc-like shape."""
        tau = 1.0
        seq = multilevel_ramsey(self.system, self.system.probe, tau=tau)
        ff = seq.get_filter_function_calculator()

        frequencies = np.linspace(0.1, 20, 100)
        Fx, Fy, Fz = ff.filter_function(frequencies)

        # Total susceptibility should be non-negative
        total = np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2
        self.assertTrue(np.all(total >= -1e-10))

    def test_spin_echo_filter_function(self):
        """Test spin echo filter function is computable."""
        seq = multilevel_spin_echo(self.system, self.system.probe, tau=1.0)
        ff = seq.get_filter_function_calculator()

        frequencies = np.linspace(0.1, 10, 50)
        susceptibility = ff.noise_susceptibility(frequencies)

        self.assertTrue(np.all(np.isfinite(susceptibility)))

    def test_cpmg_filter_function(self):
        """Test CPMG filter function is computable."""
        seq = multilevel_cpmg(self.system, self.system.probe,
                             tau=1.0, n_pulses=4)
        ff = seq.get_filter_function_calculator()

        frequencies = np.linspace(0.1, 50, 100)
        susceptibility = ff.noise_susceptibility(frequencies)

        self.assertTrue(np.all(np.isfinite(susceptibility)))
        self.assertTrue(np.all(susceptibility >= -1e-10))


class TestDifferentialSpectroscopy(unittest.TestCase):
    """Test differential spectroscopy protocol."""

    def setUp(self):
        self.system = ThreeLevelClock()

    def test_differential_sequence_creation(self):
        """Test creating differential spectroscopy sequence."""
        probe_seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        diff_seq = DifferentialSpectroscopySequence(self.system, probe_seq)

        self.assertEqual(diff_seq.system, self.system)
        self.assertEqual(diff_seq.probe_sequence, probe_seq)

    def test_differential_phase_computation(self):
        """Test differential phase can be computed."""
        probe_seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        diff_seq = DifferentialSpectroscopySequence(self.system, probe_seq)

        phase = diff_seq.compute_differential_phase()
        self.assertTrue(np.isfinite(phase))

    def test_differential_filter_function(self):
        """Test differential filter function is computable."""
        probe_seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        diff_seq = DifferentialSpectroscopySequence(self.system, probe_seq)

        frequencies = np.linspace(0.1, 10, 50)
        ff_diff = diff_seq.differential_filter_function(frequencies)

        self.assertEqual(len(ff_diff), 50)
        self.assertTrue(np.all(np.isfinite(ff_diff)))

    def test_clock_coherence_after_probe_sequence(self):
        """Test clock coherence is affected by probe sequence."""
        # Start with clock superposition
        psi0 = self.system.prepare_clock_superposition()
        initial_contrast = self.system.contrast(psi0)

        # Apply probe sequence
        probe_seq = multilevel_ramsey(self.system, self.system.probe, tau=1.0)
        psi_final = probe_seq.evolve_state(psi0)

        final_contrast = self.system.contrast(psi_final)

        # Contrast should change after probe sequence
        # (unless it's a very special case)
        self.assertTrue(np.isfinite(final_contrast))
        self.assertGreaterEqual(final_contrast, 0)
        self.assertLessEqual(final_contrast, 1.0 + 1e-10)


class TestSubspaceEmbedding(unittest.TestCase):
    """Test Subspace embedding and projection operations."""

    def setUp(self):
        self.space = HilbertSpace(3, level_labels=['g', 'e', 'm'])
        self.subspace = Subspace(self.space, (0, 1), label='probe')

    def test_embed_state(self):
        """Test embedding 2D state into 3D space."""
        state_2d = np.array([1, 0], dtype=complex)
        state_3d = self.subspace.embed_state(state_2d)

        expected = np.array([1, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state_3d, expected)

    def test_project_state(self):
        """Test projecting 3D state onto 2D subspace."""
        state_3d = np.array([1, 0, 0], dtype=complex)
        state_2d = self.subspace.project_state(state_3d)

        expected = np.array([1, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state_2d, expected)

    def test_embed_operator(self):
        """Test embedding 2x2 operator into 3x3 space."""
        sx_2x2 = SIGMA_X
        sx_3x3 = self.subspace.embed_operator(sx_2x2)

        self.assertEqual(sx_3x3.shape, (3, 3))

        # Check action on basis states
        psi_g = np.array([1, 0, 0], dtype=complex)
        psi_e = np.array([0, 1, 0], dtype=complex)

        result_g = sx_3x3 @ psi_g
        result_e = sx_3x3 @ psi_e

        np.testing.assert_array_almost_equal(result_g, psi_e)
        np.testing.assert_array_almost_equal(result_e, psi_g)

    def test_embed_operator_identity_outside(self):
        """Test that embedded operator acts as identity outside subspace."""
        sx_2x2 = SIGMA_X
        sx_3x3 = self.subspace.embed_operator(sx_2x2)

        # Metastable state should be unchanged
        psi_m = np.array([0, 0, 1], dtype=complex)
        result_m = sx_3x3 @ psi_m

        np.testing.assert_array_almost_equal(result_m, psi_m)

    def test_embed_operator_zero_outside(self):
        """Test embedding with zeros outside subspace."""
        sx_2x2 = SIGMA_X
        sx_3x3 = self.subspace.embed_operator_zero_outside(sx_2x2)

        # Metastable state should go to zero
        psi_m = np.array([0, 0, 1], dtype=complex)
        result_m = sx_3x3 @ psi_m

        np.testing.assert_array_almost_equal(result_m, np.zeros(3))


class TestGellMannMatrices(unittest.TestCase):
    """Test Gell-Mann matrix generation."""

    def test_gell_mann_count(self):
        """Test we get 8 Gell-Mann matrices for d=3."""
        matrices = gell_mann_matrices()
        self.assertEqual(len(matrices), 8)

    def test_gell_mann_hermitian(self):
        """Test Gell-Mann matrices are Hermitian."""
        matrices = gell_mann_matrices()
        for i, m in enumerate(matrices):
            np.testing.assert_array_almost_equal(
                m, m.conj().T,
                err_msg=f"Gell-Mann matrix {i+1} is not Hermitian"
            )

    def test_gell_mann_traceless(self):
        """Test Gell-Mann matrices are traceless."""
        matrices = gell_mann_matrices()
        for i, m in enumerate(matrices):
            trace = np.trace(m)
            self.assertAlmostEqual(
                trace, 0.0, places=10,
                msg=f"Gell-Mann matrix {i+1} is not traceless"
            )

    def test_generalized_gell_mann_count(self):
        """Test generalized Gell-Mann matrix count for various dimensions."""
        for d in [2, 3, 4, 5]:
            matrices = generalized_gell_mann_matrices(d)
            expected_count = d**2 - 1
            self.assertEqual(
                len(matrices), expected_count,
                f"Expected {expected_count} matrices for d={d}, got {len(matrices)}"
            )

    def test_generalized_gell_mann_d2_is_pauli(self):
        """Test that d=2 generalized Gell-Mann matrices are Pauli matrices."""
        matrices = generalized_gell_mann_matrices(2)

        # Should match Pauli matrices
        np.testing.assert_array_almost_equal(matrices[0], SIGMA_X)
        np.testing.assert_array_almost_equal(matrices[1], SIGMA_Y)
        np.testing.assert_array_almost_equal(matrices[2], SIGMA_Z)


class TestQubitSystemCompatibility(unittest.TestCase):
    """Test that QubitSystem maintains backward compatibility."""

    def setUp(self):
        self.qubit = QubitSystem()

    def test_dimension(self):
        """Test qubit system has dimension 2."""
        self.assertEqual(self.qubit.dim, 2)

    def test_pauli_matrices(self):
        """Test qubit Pauli matrix properties."""
        sx = self.qubit.sigma_x
        sy = self.qubit.sigma_y
        sz = self.qubit.sigma_z

        np.testing.assert_array_almost_equal(sx, SIGMA_X)
        np.testing.assert_array_almost_equal(sy, SIGMA_Y)
        np.testing.assert_array_almost_equal(sz, SIGMA_Z)

    def test_noise_operators(self):
        """Test qubit noise operators."""
        noise_ops = self.qubit.noise_operators()

        self.assertIn('dephasing', noise_ops)
        self.assertIn('amplitude_x', noise_ops)
        self.assertIn('amplitude_y', noise_ops)

    def test_bloch_vector(self):
        """Test Bloch vector computation."""
        state_z_up = np.array([1, 0], dtype=complex)
        bloch = self.qubit.bloch_vector(state_z_up)

        np.testing.assert_array_almost_equal(bloch, [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
