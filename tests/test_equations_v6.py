"""v0.0006 regression tests for the entanglement, symmetry and measurement layers.

Every new opcode is checked against an independent closed-form reference (the
exact classical reduction), and each layer's defining invariant is asserted.
The IV completion (`effective_action`) is checked against the Gaussian
Legendre identity Gamma(phi(J)) = W[J].
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as anp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvm import *  # noqa: F401,F403

from qvm.ir import Program
from qvm.backends.pennylane_standard import UnsupportedLowering

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.diag([1, -1]).astype(complex)


def random_ket(rng, dim=4):
    vector = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


def bell_state():
    ket = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return ket, np.outer(ket, ket.conj())


class EntanglementLayerTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()
        self.rng = np.random.default_rng(6)

    def test_tensor_product_matches_kron_and_preserves_norm(self):
        left = random_ket(self.rng, 2)
        right = random_ket(self.rng, 2)
        joint = np.asarray(self.vm.run(tensor_product_program(),
                                       {'left': left, 'right': right},
                                       backend='numpy_overlap'))
        np.testing.assert_allclose(joint, np.kron(left, right), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(joint)), 1.0, places=12)

    def test_tensor_product_is_batch_aware(self):
        left = np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
        right = np.stack([np.array([0.0, 1.0]), np.array([1.0, 0.0])])
        joint = np.asarray(self.vm.run(tensor_product_program(),
                                       {'left': left, 'right': right},
                                       backend='numpy_overlap'))
        self.assertEqual(joint.shape, (2, 4))
        np.testing.assert_allclose(joint[0], np.kron(left[0], right[0]), atol=1e-12)
        np.testing.assert_allclose(joint[1], np.kron(left[1], right[1]), atol=1e-12)

    def test_partial_trace_preserves_trace_and_reduces_bell_to_mixed(self):
        _, rho = bell_state()
        reduced = np.asarray(self.vm.run(partial_trace_program((2, 2)), {'state': rho},
                                         backend='numpy_overlap'))
        np.testing.assert_allclose(reduced, np.eye(2) / 2, atol=1e-12)
        self.assertAlmostEqual(float(np.trace(reduced)), 1.0, places=12)

    def test_partial_trace_of_product_state_is_pure(self):
        left = np.array([1.0, 0.0])
        right = np.array([0.0, 1.0])
        ket = np.kron(left, right)
        rho = np.outer(ket, ket.conj())
        reduced = np.asarray(self.vm.run(partial_trace_program((2, 2)), {'state': rho},
                                         backend='numpy_overlap'))
        np.testing.assert_allclose(reduced, np.outer(left, left.conj()), atol=1e-12)

    def test_partial_trace_rejects_dimension_mismatch(self):
        _, rho = bell_state()
        with self.assertRaises(ValueError):
            self.vm.run(partial_trace_program((2, 3)), {'state': rho},
                        backend='numpy_overlap')

    def test_schmidt_spectrum_squared_sums_to_one(self):
        ket = random_ket(self.rng)
        rho = np.outer(ket, ket.conj())
        coefficients = np.asarray(self.vm.run(schmidt_spectrum_program((2, 2)),
                                              {'state': rho}, backend='numpy_overlap'))
        reference = np.linalg.svd(ket.reshape(2, 2), compute_uv=False)
        np.testing.assert_allclose(np.sort(coefficients), np.sort(reference), atol=1e-12)
        self.assertAlmostEqual(float(np.sum(coefficients ** 2)), 1.0, places=12)

    def test_entanglement_entropy_extremes(self):
        bell_ket, bell_rho = bell_state()
        entropy = float(self.vm.run(entanglement_entropy_program((2, 2)),
                                    {'state': bell_rho}, backend='numpy_overlap'))
        self.assertAlmostEqual(entropy, 1.0, places=12)
        product = np.kron(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        rho = np.outer(product, product.conj())
        entropy = float(self.vm.run(entanglement_entropy_program((2, 2)),
                                    {'state': rho}, backend='numpy_overlap'))
        self.assertAlmostEqual(entropy, 0.0, places=12)

    def test_swap_test_matches_squared_overlap(self):
        left = random_ket(self.rng, 2)
        right = random_ket(self.rng, 2)
        fidelity = float(self.vm.run(swap_test_program(), {'left': left, 'right': right},
                                     backend='numpy_overlap'))
        self.assertAlmostEqual(fidelity, abs(np.vdot(left, right)) ** 2, places=12)
        same = float(self.vm.run(swap_test_program(), {'left': left, 'right': left},
                                 backend='numpy_overlap'))
        self.assertAlmostEqual(same, 1.0, places=12)
        orthogonal = float(self.vm.run(swap_test_program(),
                                       {'left': np.array([1.0, 0.0]),
                                        'right': np.array([0.0, 1.0])},
                                       backend='numpy_overlap'))
        self.assertAlmostEqual(orthogonal, 0.0, places=12)

    def test_swap_test_gradient_flows_through_amplitudes(self):
        p = Program('swap_gradient')
        p.input('left', 'amplitude')
        p.input('right', 'amplitude')
        p.emit('fidelity', 'swap_test', 'left', 'right', result_type='real')
        p.returns('fidelity')
        fixed = anp.array([0.8, -0.6])

        def loss(left):
            return self.vm.run(p, {'left': left, 'right': fixed}, backend='numpy_overlap')

        left = anp.array([0.6, 0.8], requires_grad=True)
        gradient = qml.grad(loss)(left)
        # d |<r|l>|^2 / dl = 2 <l|r> * conj(r) componentwise.
        expected = 2 * np.vdot(left, fixed) * np.conj(fixed)
        np.testing.assert_allclose(np.asarray(gradient), expected, atol=1e-9)


class SymmetryLayerTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()
        self.rng = np.random.default_rng(7)

    def test_symmetry_generator_builds_pauli_strings(self):
        generator = np.asarray(self.vm.run(symmetry_generator_program('zz'),
                                           {}, backend='numpy_overlap'))
        np.testing.assert_allclose(generator, np.kron(PAULI_Z, PAULI_Z), atol=1e-12)
        self.assertTrue(np.allclose(generator, generator.conj().T))

    def test_symmetry_generator_eigenvalues_are_plus_minus_one(self):
        for label in ('x', 'y', 'z', 'zx', 'xyz', 'iiz'):
            generator = np.asarray(self.vm.run(symmetry_generator_program(label),
                                               {}, backend='numpy_overlap'))
            eigenvalues = np.linalg.eigvalsh(generator)
            np.testing.assert_allclose(np.abs(eigenvalues), 1.0, atol=1e-12)

    def test_symmetry_generator_rejects_bad_labels(self):
        program = symmetry_generator_program('a')
        with self.assertRaises(UnsupportedLowering):
            self.vm.run(program, {}, backend='numpy_overlap')

    def test_irrep_projector_is_an_orthogonal_projection(self):
        generator = np.diag([1.0, 1.0, 2.0])
        projector = np.asarray(self.vm.run(irrep_projector_program(1.0),
                                           {'generator': generator},
                                           backend='numpy_overlap'))
        np.testing.assert_allclose(projector, np.diag([1.0, 1.0, 0.0]), atol=1e-12)
        self.assertTrue(np.allclose(projector @ projector, projector, atol=1e-12))
        self.assertTrue(np.allclose(projector, projector.conj().T, atol=1e-12))

    def test_irrep_projectors_resolve_the_identity(self):
        generator = np.array([[0.0, 1.0], [1.0, 0.0]])
        total = np.zeros((2, 2))
        for eigenvalue in (1.0, -1.0):
            total += np.asarray(self.vm.run(irrep_projector_program(eigenvalue),
                                            {'generator': generator},
                                            backend='numpy_overlap'))
        np.testing.assert_allclose(total, np.eye(2), atol=1e-12)

    def test_conserved_current_matches_expectation(self):
        ket = random_ket(self.rng, 2)
        charge = np.array([[0.5, 0.3], [0.3, -0.5]])
        current = float(self.vm.run(conserved_current_program(),
                                    {'state': ket, 'charge': charge},
                                    backend='numpy_overlap'))
        self.assertAlmostEqual(current, float(np.real(np.vdot(ket, charge @ ket))),
                               places=12)

    def test_conservation_defect_zero_for_commuting_generators(self):
        defect = float(self.vm.run(conservation_defect_program(),
                                   {'left': np.diag([1.0, 2.0]),
                                    'right': np.diag([3.0, -1.0])},
                                   backend='numpy_overlap'))
        self.assertAlmostEqual(defect, 0.0, places=12)

    def test_conservation_defect_equals_commutator_strength(self):
        left = np.diag([1.0, 2.0])
        right = np.array([[0.0, 1.0], [1.0, 0.0]])
        defect = float(self.vm.run(conservation_defect_program(),
                                   {'left': left, 'right': right},
                                   backend='numpy_overlap'))
        strength = float(self.vm.run(commutator_strength(),
                                     {'left': left, 'right': right},
                                     backend='numpy_overlap'))
        self.assertAlmostEqual(defect, strength, places=12)


class MeasurementLayerTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()
        self.rng = np.random.default_rng(8)
        ket = np.array([np.cos(0.4), np.sin(0.4)])
        self.rho = np.outer(ket, ket.conj())

    def test_identity_kraus_set_leaves_the_state_unchanged(self):
        kraus = np.eye(2)[None, :, :]
        updated = np.asarray(self.vm.run(instrument_channel_program(),
                                         {'state': self.rho, 'kraus': kraus},
                                         backend='numpy_overlap'))
        np.testing.assert_allclose(updated, self.rho, atol=1e-12)

    def test_dephasing_kraus_set_removes_coherence(self):
        kraus = np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
        updated = np.asarray(self.vm.run(instrument_channel_program(),
                                         {'state': self.rho, 'kraus': kraus},
                                         backend='numpy_overlap'))
        np.testing.assert_allclose(updated, np.diag(np.diag(self.rho)), atol=1e-12)
        self.assertAlmostEqual(float(np.trace(updated)), 1.0, places=12)

    def test_kraus_apply_is_batch_aware(self):
        ket = np.array([np.cos(0.9), np.sin(0.9)])
        other = np.outer(ket, ket.conj())
        batch = np.stack([self.rho, other])
        kraus = np.eye(2)[None, :, :]
        updated = np.asarray(self.vm.run(instrument_channel_program(),
                                         {'state': batch, 'kraus': kraus},
                                         backend='numpy_overlap'))
        self.assertEqual(updated.shape, (2, 2, 2))
        np.testing.assert_allclose(updated[0], self.rho, atol=1e-12)
        np.testing.assert_allclose(updated[1], other, atol=1e-12)

    def test_povm_probabilities_sum_to_one_and_match_reference(self):
        effects = np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
        probabilities = np.asarray(self.vm.run(povm_probability_program(),
                                               {'state': self.rho, 'effects': effects},
                                               backend='numpy_overlap'))
        reference = np.real(np.einsum('ab,kba->k', self.rho, effects))
        np.testing.assert_allclose(probabilities, reference, atol=1e-12)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)

    def test_postselect_collapses_to_the_conditioned_eigenstate(self):
        ket = np.array([np.cos(0.7), np.sin(0.7)])
        mixed = 0.5 * np.outer(ket, ket.conj()) + 0.5 * np.diag([0.3, 0.7])
        effect = np.diag([1.0, 0.0])
        collapsed = np.asarray(self.vm.run(postselect_program(),
                                           {'state': mixed, 'effect': effect},
                                           backend='numpy_overlap'))
        np.testing.assert_allclose(collapsed, np.diag([1.0, 0.0]), atol=1e-12)
        self.assertAlmostEqual(float(np.trace(collapsed)), 1.0, places=12)

    def test_postselect_rejects_zero_probability(self):
        effect = np.diag([1.0, 0.0])
        rho = np.diag([0.0, 1.0])
        with self.assertRaises(NumericalDomainError):
            self.vm.run(postselect_program(), {'state': rho, 'effect': effect},
                        backend='numpy_overlap')


class SchwingerCompletionTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()

    def test_effective_action_matches_solve_reference(self):
        kernel = np.array([[2.0, 0.5], [0.5, 3.0]])
        field = np.array([0.7, -0.3])
        action = float(self.vm.run(effective_action_program(),
                                   {'kernel': kernel, 'mean_field': field},
                                   backend='numpy_overlap'))
        solved = np.linalg.solve(kernel, field)
        self.assertAlmostEqual(action, 0.5 * float(np.real(np.vdot(field, solved))),
                               places=12)

    def test_legendre_identity_gamma_of_mean_field_equals_generating_functional(self):
        generator = np.array([[1.0, 0.2], [0.2, -0.5]])
        omega = -2.0
        source = np.array([0.7, -0.3])
        propagator = np.asarray(self.vm.run(resolvent_program(),
                                            {'generator': generator, 'source_energy': omega},
                                            backend='numpy_overlap'))
        action = float(self.vm.run(generating_functional_program(),
                                   {'propagator': propagator, 'source': source},
                                   backend='numpy_overlap'))
        mean_field = propagator @ source
        effective = float(self.vm.run(effective_action_program(),
                                      {'kernel': propagator,
                                       'mean_field': mean_field},
                                      backend='numpy_overlap'))
        self.assertAlmostEqual(effective, action, places=12)


class SchemaV6Tests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()

    def test_new_programs_round_trip_and_reseal_as_current(self):
        for program in (tensor_product_program(), partial_trace_program((2, 2)),
                        entanglement_entropy_program((2, 2)), swap_test_program(),
                        symmetry_generator_program('zz'), irrep_projector_program(1.0),
                        instrument_channel_program(), postselect_program(),
                        effective_action_program()):
            restored = Program.from_json(program.to_json())
            self.assertEqual(restored.record()['schema'], 'qvm_v0.0006')
            self.assertEqual(restored.sha256, program.sha256)
            self.assertTrue(restored.sealed)

    def test_v5_json_is_imported_then_resealed_as_current(self):
        text = swap_test_program().to_json().replace('qvm_v0.0006', 'qvm_v0.0005')
        restored = Program.from_json(text)
        self.assertEqual(restored.record()['schema'], 'qvm_v0.0006')
        self.assertTrue(restored.sealed)

    def test_result_types_are_part_of_the_tape(self):
        record = entanglement_entropy_program((2, 2)).record()
        self.assertEqual(record['instructions'][-1]['result_type'], 'real')
        record = instrument_channel_program().record()
        self.assertEqual(record['instructions'][-1]['result_type'], 'density')
        record = schmidt_spectrum_program((2, 2)).record()
        self.assertEqual(record['instructions'][-1]['result_type'], 'spectrum')

    def test_invalid_dims_are_rejected_before_execution(self):
        program = partial_trace_program((2,))
        with self.assertRaises(UnsupportedLowering):
            self.vm.run(program, {'state': np.eye(4) / 4}, backend='numpy_overlap')


if __name__ == '__main__':
    unittest.main()
