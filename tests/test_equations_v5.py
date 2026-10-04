"""v0.0005 regression tests for the four quantum-equation layers.

Every new opcode is checked against an independent closed-form reference (the
exact classical reduction), and each layer's defining invariant is asserted.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as anp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvm import *  # noqa: F401,F403

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.diag([1, -1]).astype(complex)


def random_ket(rng):
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    return vector / np.linalg.norm(vector)


class OverlapLayerTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()
        self.rng = np.random.default_rng(5)

    def _gram(self, angles):
        flat = np.asarray(angles).reshape(-1)
        states = np.stack([np.array([np.cos(a / 2), np.sin(a / 2)]) for a in flat])
        return states.conj() @ states.T, states

    def test_gram_spectrum_and_participation_match_reference(self):
        angles = np.array([[0.0], [0.4], [1.2], [2.1]])
        gram, _ = self._gram(angles)
        spectrum = np.asarray(self.vm.run(overlap_spectrum(), {'angles': angles},
                                          backend='numpy_overlap'))
        np.testing.assert_allclose(spectrum, np.linalg.eigvalsh(gram), atol=1e-12)
        participation = float(np.asarray(self.vm.run(
            overlap_participation(), {'angles': angles}, backend='numpy_overlap')))
        reference = spectrum.sum() ** 2 / (spectrum ** 2).sum()
        self.assertAlmostEqual(participation, reference, places=12)

    def test_phase_ablation_drops_coherence_and_keeps_diagonal(self):
        angles = np.array([[0.0], [0.4], [1.2]])
        gram, _ = self._gram(angles)
        coherence = float(np.asarray(self.vm.run(overlap_coherence(), {'angles': angles},
                                                 backend='numpy_overlap')))
        np.testing.assert_allclose(coherence, np.linalg.norm(
            gram - np.diag(np.diag(gram))), atol=1e-12)
        ablated = np.asarray(self.vm.run(phase_ablated_gram(), {'angles': angles},
                                        backend='numpy_overlap'))
        np.testing.assert_allclose(ablated, np.diag(np.diag(gram)), atol=1e-12)
        np.testing.assert_allclose(np.diag(ablated), np.diag(gram), atol=1e-12)
        offdiagonal = ablated - np.diag(np.diag(ablated))
        np.testing.assert_allclose(offdiagonal, 0, atol=1e-12)
        self.assertGreater(coherence, 0)

    def test_orthogonal_states_have_unit_participation_and_zero_coherence(self):
        angles = np.array([[0.0], [np.pi]])
        participation = float(np.asarray(self.vm.run(
            overlap_participation(), {'angles': angles}, backend='numpy_overlap')))
        self.assertAlmostEqual(participation, 2.0, places=10)
        coherence = float(np.asarray(self.vm.run(overlap_coherence(), {'angles': angles},
                                                 backend='numpy_overlap')))
        self.assertLess(coherence, 1e-12)


class SchrodingerLayerTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()

    def test_matrix_exponential_is_unitary_and_matches_expm(self):
        generator = np.array([[0.7, 0.3 - 0.2j], [0.3 + 0.2j, -1.1]])
        time = 0.9
        unitary = np.asarray(self.vm.run(generator_evolution(),
                                         {'generator': generator, 'time': time},
                                         backend='numpy_overlap'))
        eigenvalues, eigenvectors = np.linalg.eigh(generator)
        reference = (eigenvectors * np.exp(-1j * time * eigenvalues)) @ eigenvectors.conj().T
        np.testing.assert_allclose(unitary, reference, atol=1e-12)
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-12)

    def test_generator_spectrum_matches_eigvalsh(self):
        generator = np.array([[2.0, 0.4], [0.4, -0.5]])
        spectrum = np.asarray(self.vm.run(generator_spectrum(), {'generator': generator},
                                          backend='numpy_overlap'))
        np.testing.assert_allclose(spectrum, np.linalg.eigvalsh(generator), atol=1e-12)

    def test_commutator_strength_matches_frobenius_norm(self):
        left = np.diag([1.0, 2.0])
        right = np.array([[0.0, 1.0], [1.0, 0.0]])
        strength = float(np.asarray(self.vm.run(
            commutator_strength(), {'left': left, 'right': right}, backend='numpy_overlap')))
        self.assertAlmostEqual(strength, np.linalg.norm(left @ right - right @ left), places=12)

    def test_time_ordering_is_order_sensitive(self):
        generators = np.stack([np.diag([1.0, 2.0]), np.array([[0.0, 1.0], [1.0, 0.0]])])
        times = np.array([0.6, 0.9])
        u2 = np.asarray(self.vm.run(generator_evolution(),
                                    {'generator': generators[1], 'time': times[1]},
                                    backend='numpy_overlap'))
        u1 = np.asarray(self.vm.run(generator_evolution(),
                                    {'generator': generators[0], 'time': times[0]},
                                    backend='numpy_overlap'))
        forward = np.asarray(self.vm.run(time_ordered_sequence(),
                                         {'generators': generators, 'times': times},
                                         backend='numpy_overlap'))
        np.testing.assert_allclose(forward, u2 @ u1, atol=1e-12)
        reversed_ = np.asarray(self.vm.run(time_ordered_sequence(),
                                           {'generators': generators[::-1], 'times': times[::-1]},
                                           backend='numpy_overlap'))
        self.assertGreater(np.linalg.norm(forward - reversed_), 0.1)


class DualityLayerTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()
        self.rng = np.random.default_rng(7)

    def test_detector_duality_extremes_and_unit_relation(self):
        for _ in range(5):
            left, right = random_ket(self.rng), random_ket(self.rng)
            pair = np.asarray(self.vm.run(detector_duality_program(),
                                          {'left': left, 'right': right},
                                          backend='numpy_overlap'))
            self.assertAlmostEqual(pair[0] ** 2 + pair[1] ** 2, 1.0, places=12)
            self.assertAlmostEqual(pair[1], abs(np.vdot(left, right)), places=12)
        orthogonal = np.asarray(self.vm.run(detector_duality_program(),
                                           {'left': np.array([1, 0]), 'right': np.array([0, 1])},
                                           backend='numpy_overlap'))
        np.testing.assert_allclose(orthogonal, [1.0, 0.0], atol=1e-12)

    def test_path_duality_pure_equals_one_and_dephasing_shrinks_visibility(self):
        theta = 0.8
        state = np.array([np.cos(theta / 2), np.sin(theta / 2)])
        rho = np.outer(state, state.conj())
        pair = np.asarray(self.vm.run(path_duality_program(), {'state': rho},
                                      backend='numpy_overlap'))
        self.assertAlmostEqual(pair[0] ** 2 + pair[1] ** 2, 1.0, places=12)
        self.assertAlmostEqual(pair[1], 2 * abs(state[0] * state[1]), places=12)
        dephased = np.diag(np.diag(rho))
        mixed = np.asarray(self.vm.run(path_duality_program(), {'state': dephased},
                                       backend='numpy_overlap'))
        self.assertAlmostEqual(mixed[1], 0.0, places=12)
        self.assertAlmostEqual(mixed[0], pair[0], places=12)

    def test_duality_slack_is_nonnegative_and_vanishes_for_pure(self):
        for a, b in ((0.5, 0.5), (0.3, 0.9), (1.0, 0.0)):
            slack = float(np.asarray(self.vm.run(duality_slack_program(),
                                                 {'pair': np.array([a, b])},
                                                 backend='numpy_overlap')))
            self.assertAlmostEqual(slack, 1 - a ** 2 - b ** 2, places=12)
            self.assertGreaterEqual(slack, -1e-12)


class SchwingerLayerTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()

    def _generator(self):
        return np.array([[1.2, 0.2], [0.2, -0.4]])

    def test_resolvent_inverts_shifted_generator(self):
        generator = self._generator()
        omega = -0.5
        propagator = np.asarray(self.vm.run(resolvent_program(),
                                            {'generator': generator, 'source_energy': omega},
                                            backend='numpy_overlap'))
        np.testing.assert_allclose((generator - omega * np.eye(2)) @ propagator, np.eye(2),
                                   atol=1e-12)

    def test_proper_time_resolvent_converges_to_resolvent(self):
        generator = self._generator()
        omega = -0.5
        exact = np.asarray(self.vm.run(resolvent_program(),
                                       {'generator': generator, 'source_energy': omega},
                                       backend='numpy_overlap'))
        approximate = np.asarray(self.vm.run(
            resolvent_program(truncated=True),
            {'generator': generator, 'source_energy': omega}, backend='numpy_overlap'))
        eigenvalues = np.linalg.eigvalsh(generator)
        smallest = float(np.min(eigenvalues - omega))
        bound = np.exp(-smallest * NumericalPolicy().proper_time_cutoff) / smallest
        self.assertLess(np.max(np.abs(approximate - exact)), bound)

    def test_source_response_and_generating_functional_match_reference(self):
        generator = self._generator()
        omega = -0.5
        propagator = np.asarray(self.vm.run(resolvent_program(),
                                            {'generator': generator, 'source_energy': omega},
                                            backend='numpy_overlap'))
        source = np.array([1.0, 0.5])
        field = np.asarray(self.vm.run(source_response_program(),
                                       {'propagator': propagator, 'source': source},
                                       backend='numpy_overlap'))
        np.testing.assert_allclose(field, propagator @ source, atol=1e-12)
        action = float(np.asarray(self.vm.run(
            generating_functional_program(),
            {'propagator': propagator, 'source': source}, backend='numpy_overlap')))
        self.assertAlmostEqual(action, 0.5 * np.real(np.vdot(source, propagator @ source)),
                               places=12)

    def test_generating_functional_gradient_is_response(self):
        generator = self._generator()
        propagator = np.asarray(self.vm.run(resolvent_program(),
                                            {'generator': generator, 'source_energy': -0.5},
                                            backend='numpy_overlap'))

        def loss(source):
            return self.vm.run(generating_functional_program(),
                               {'propagator': propagator, 'source': source},
                               backend='numpy_overlap')

        source = anp.array([0.7, -0.3], requires_grad=True)
        gradient = qml.grad(loss)(source)
        np.testing.assert_allclose(np.asarray(gradient), propagator @ np.asarray(source),
                                   atol=1e-9)


class SchemaTests(unittest.TestCase):
    def test_new_programs_round_trip_and_reseal_as_current(self):
        for program in (overlap_spectrum(), generator_evolution(), detector_duality_program(),
                        resolvent_program(truncated=True), generating_functional_program()):
            restored = Program.from_json(program.to_json())
            self.assertEqual(restored.record()['schema'], 'qvm_v0.0005')
            self.assertEqual(restored.sha256, program.sha256)
            self.assertTrue(restored.sealed)

    def test_result_types_are_part_of_the_tape(self):
        record = resolvent_program().record()
        self.assertEqual(record['instructions'][-1]['result_type'], 'operator')
        record = detector_duality_program().record()
        self.assertEqual(record['instructions'][-1]['result_type'], 'real')

    def test_domain_error_on_nonpositive_shift(self):
        vm = QVM()
        with self.assertRaises(NumericalDomainError):
            vm.run(resolvent_program(), {'generator': np.diag([0.5, 2.0]), 'source_energy': 1.0},
                   backend='numpy_overlap')


if __name__ == '__main__':
    unittest.main()