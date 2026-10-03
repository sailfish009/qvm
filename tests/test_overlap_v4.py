"""Representation, algebra, gradient, and adversarial-contract regressions."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pennylane as qml
from pennylane import numpy as anp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvm import (Program, QVM, NumericalPolicy, NumericalDomainError,
                 UnsupportedLowering, encoded_overlap, encoded_gram,
                 state_superposition, regularized_span_projection,
                 validate_value, escort_profile)


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def one_op(op, inputs, result_type, **attrs):
    p = Program(op)
    for name, kind in inputs:
        p.input(name, kind)
    p.emit('out', op, *(x[0] for x in inputs), result_type=result_type, **attrs)
    return p.returns('out')


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.vm = QVM()
        self.rng = np.random.default_rng(41)

    def run_program(self, p, inputs, backend='numpy_overlap', **kw):
        return self.vm.run(p, inputs, backend=backend, **kw)

    def test_legacy_pure_state_output_fixed(self):
        p = Program('legacy_pure_return')
        p.input('angles', 'angles', [2])
        p.emit('psi', 'ry_product_state', 'angles', role='state')
        p.returns('psi')
        psi = self.vm.run(p, {'angles': np.array([.2, .4])})
        self.assertEqual(psi.shape, (4,))
        self.assertAlmostEqual(np.linalg.norm(psi), 1.)

    def test_types_not_roles_and_normalization_not_silently_repaired(self):
        p = Program('not_density')
        p.input('s', 'state_collection')
        p.emit('g', 'gram_matrix', 's', role='state', result_type='gram')
        p.returns('g')
        g = self.run_program(p, {'s': np.eye(3)})
        self.assertEqual(np.trace(g), 3.)
        for kind, value in [('ket', [2., 0]), ('density', np.eye(2)),
                            ('gram', [[1, 2], [0, 1]]), ('gram', [[1, 2], [2, 1]]),
                            ('real', np.array([1+0j])), ('amplitude', [np.nan]),
                            ('ket', np.ones((0, 2)))]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_value(np.asarray(value), kind, NumericalPolicy())
        validate_value(np.zeros(2), 'amplitude', NumericalPolicy())

    def test_batched_encoders_and_raw_overlap_pennylane(self):
        for complex_encoding in (False, True):
            shape = (2, 3, 2, 2) if complex_encoding else (2, 3, 2)
            left = self.rng.normal(size=shape)
            right = self.rng.normal(size=shape)
            for fidelity in (False, True):
                p = encoded_overlap(complex_encoding, fidelity)
                inputs = {'left': left, 'right': right}
                a = self.run_program(p, inputs)
                b = self.run_program(p, inputs, 'pennylane_overlap')
                self.assertEqual(a.shape, (2, 3))
                np.testing.assert_allclose(a, b, atol=2e-14, rtol=0)

    def test_product_ry_historical_equation(self):
        # Exact formula endpoint, not a claim of retraining historical checkpoints.
        left, right = self.rng.normal(size=(2, 4, 12))
        f = self.run_program(encoded_overlap(fidelity=True), {'left': left, 'right': right})
        np.testing.assert_allclose(f, np.prod(np.cos((left-right)/2)**2, axis=-1), atol=1e-14)

    def test_gram_spectrum_and_cyclic_gauge_invariance(self):
        states = normalize(self.rng.normal(size=(2, 4, 3)) + 1j*self.rng.normal(size=(2, 4, 3)))
        p = one_op('gram_matrix', [('s', 'state_collection')], 'gram')
        g = self.run_program(p, {'s': states})
        np.testing.assert_allclose(g, np.einsum('...id,...jd->...ij', states.conj(), states))
        np.testing.assert_allclose(np.diagonal(g, axis1=-2, axis2=-1), 1)
        self.assertGreater(np.linalg.eigvalsh(g).min(), -1e-12)
        phases = np.exp(1j*self.rng.normal(size=(2, 4, 1)))
        h = self.run_program(p, {'s': states*phases})
        self.assertGreater(np.max(abs(g-h)), .01)
        cycle = one_op('cyclic_overlap', [('g', 'gram')], 'complex', indices=[0, 1, 2])
        np.testing.assert_allclose(self.run_program(cycle, {'g': g}), self.run_program(cycle, {'g': h}), atol=1e-14)

    def test_superposition_cross_terms_and_zero_branch(self):
        states = normalize(self.rng.normal(size=(3, 4)) + 1j*self.rng.normal(size=(3, 4)))
        coefficients = self.rng.normal(size=3) + 1j*self.rng.normal(size=3)
        u = self.run_program(state_superposition(False), {'states': states, 'coefficients': coefficients})
        g = states.conj() @ states.T
        self.assertAlmostEqual(float(np.vdot(u, u).real), float(np.vdot(coefficients, g@coefficients).real))
        normed = self.run_program(state_superposition(), {'states': states, 'coefficients': coefficients})
        np.testing.assert_allclose(normed, u/np.linalg.norm(u))
        with self.assertRaises(NumericalDomainError):
            self.run_program(state_superposition(), {'states': np.array([[1., 0], [1., 0]]), 'coefficients': np.array([1., -1.])})

    def test_extreme_amplitude_normalization(self):
        p = one_op('normalize', [('a', 'amplitude')], 'ket')
        base = np.array([1.+1j, 2.-1j])
        for scale in (1e-300, 1e300):
            actual = self.run_program(p, {'a': scale*base})
            np.testing.assert_allclose(actual, normalize(base), atol=1e-15)
        huge=np.array([complex(1.7e308,1.7e308),complex(-1.7e308,1.7e308)])
        np.testing.assert_allclose(self.run_program(p,{'a':huge}),np.array([1+1j,-1+1j])/2,atol=1e-15)

    def test_density_evolution_expectation_and_complex_conjugation(self):
        state = normalize(np.array([1+2j, 3-1j]))
        unitary = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
        evolved = self.run_program(one_op('apply_unitary', [('s','ket'), ('u','unitary')], 'ket'), {'s':state, 'u':unitary})
        np.testing.assert_allclose(evolved, unitary@state)
        rho = self.run_program(one_op('ket_density', [('s','ket')], 'density'), {'s':evolved})
        np.testing.assert_allclose(rho, np.outer(evolved, evolved.conj()))
        y = np.array([[0, -1j], [1j, 0]])
        result = self.run_program(one_op('expectation', [('s','ket'), ('o','hermitian')], 'real'), {'s':evolved, 'o':y})
        self.assertAlmostEqual(float(result), float(np.trace(rho@y).real))
        z = self.run_program(one_op('inner_product', [('a','ket'), ('b','ket')], 'complex'), {'a':state, 'b':evolved})
        np.testing.assert_allclose(z, np.vdot(state, evolved))

    def test_batched_unitary_input_and_readout(self):
        states = normalize(self.rng.normal(size=(3, 2)) + 1j*self.rng.normal(size=(3, 2)))
        u = np.broadcast_to(np.eye(2), (3, 2, 2))
        p = one_op('apply_unitary', [('s','ket'), ('u','unitary')], 'ket')
        np.testing.assert_allclose(self.run_program(p, {'s':states, 'u':u}), states)

    def test_ridge_rank_deficiency_reference_and_audit(self):
        states = np.array([[1., 0], [1., 0], [0, 1.]])
        state = normalize(np.array([1., 2.]))
        p = regularized_span_projection(.1)
        result = self.run_program(p, {'states':states, 'state':state}, audit=True)
        expected = states.T @ np.linalg.solve(states@states.T+.1*np.eye(3), states@state)
        np.testing.assert_allclose(result.value, expected)
        self.assertEqual(result.numerical_interventions[0]['kind'], 'declared_ridge_regularization')
        self.assertFalse(result.numerical_interventions[0]['exact_projector'])
        self.assertGreater(np.linalg.norm(result.value-state), 0)
        phases = np.exp(1j*np.array([.2, -.7, .8]))
        transformed = self.run_program(p, {'states':states*phases[:,None], 'state':state})
        np.testing.assert_allclose(transformed, expected, atol=1e-14)
        with self.assertRaises(UnsupportedLowering):
            self.run_program(regularized_span_projection(0), {'states':states, 'state':state})

    def test_directional_gradients_each_backend(self):
        p = encoded_overlap(complex_encoding=True, fidelity=True)
        left = anp.array(self.rng.normal(size=(2, 2, 2)), requires_grad=True)
        right = self.rng.normal(size=(2, 2, 2))
        direction = self.rng.normal(size=left.shape)
        gradients = []
        for backend in ('numpy_overlap', 'pennylane_overlap'):
            def loss(x):
                return anp.sum(self.run_program(p, {'left':x, 'right':right}, backend))
            gradient = qml.grad(loss)(left)
            eps = 1e-6
            fd = (loss(left+eps*direction)-loss(left-eps*direction))/(2*eps)
            self.assertAlmostEqual(float(anp.sum(gradient*direction)), float(fd), places=7)
            gradients.append(gradient)
        np.testing.assert_allclose(*gradients, atol=1e-12, rtol=0)

    def test_composed_gram_ridge_gradient(self):
        p = Program('trainable_span')
        p.input('a','angles'); p.input('s','ket')
        p.emit('k','encode_ryrz','a',result_type='ket')
        p.emit('c','as_collection','k',result_type='state_collection')
        p.emit('u','ridge_project','c','s',ridge=.2,result_type='amplitude')
        p.emit('z','inner_product','u','u',result_type='complex')
        p.emit('f','absolute_square','z',result_type='real'); p.returns('f')
        a = anp.array(self.rng.normal(size=(2,3,2,2)), requires_grad=True)
        s = normalize(self.rng.normal(size=(2,4)) + 1j*self.rng.normal(size=(2,4)))
        d = self.rng.normal(size=a.shape)
        for backend in ('numpy_overlap','pennylane_overlap'):
            loss = lambda x: anp.sum(self.run_program(p, {'a':x,'s':s},backend))
            grad = qml.grad(loss)(a)
            eps=1e-6
            fd=(loss(a+eps*d)-loss(a-eps*d))/(2*eps)
            self.assertAlmostEqual(float(anp.sum(grad*d)),float(fd),places=6)

    def test_superposition_normalize_readout_gradient(self):
        p = Program('coefficient_learning')
        p.input('c','coefficients'); p.input('s','state_collection'); p.input('o','hermitian')
        p.emit('u','linear_combination','s','c',result_type='amplitude')
        p.emit('v','normalize','u',result_type='ket')
        p.emit('r','expectation','v','o',result_type='real'); p.returns('r')
        states = normalize(self.rng.normal(size=(3,2))+1j*self.rng.normal(size=(3,2)))
        c = anp.array([.2,.6,-.1],requires_grad=True)
        def loss(t):
            coeff = t*anp.exp(1j*t)
            return self.run_program(p, {'c':coeff, 's':states, 'o':np.diag([1.,-1.])})
        g=qml.grad(loss)(c); e=1e-6
        for i in range(3):
            step=np.eye(3)[i]*e
            self.assertAlmostEqual(float(g[i]),float((loss(c+step)-loss(c-step))/(2*e)),places=7)

    def test_density_cycle_and_evolution_gradients(self):
        p=Program('cycle_gradient')
        p.input('a','angles')
        p.emit('s','encode_ryrz','a',result_type='ket')
        p.emit('c','as_collection','s',result_type='state_collection')
        p.emit('g','gram_matrix','c',result_type='gram')
        p.emit('z','cyclic_overlap','g',indices=[0,1,2],result_type='complex');p.returns('z')
        a=anp.array(self.rng.normal(size=(3,2,2)),requires_grad=True)
        d=self.rng.normal(size=a.shape)
        for backend in ('numpy_overlap','pennylane_overlap'):
            loss=lambda t: anp.real(self.run_program(p,{'a':t},backend)) + .7*anp.imag(self.run_program(p,{'a':t},backend))
            g=qml.grad(loss)(a); e=1e-6
            self.assertAlmostEqual(float(anp.sum(g*d)),float((loss(a+e*d)-loss(a-e*d))/(2*e)),places=7)
        p=Program('evolved_density')
        p.input('a','angles');p.input('u','unitary')
        p.emit('s','encode_ryrz','a',result_type='ket')
        p.emit('v','apply_unitary','s','u',result_type='ket')
        p.emit('r','ket_density','v',result_type='density');p.returns('r')
        u=np.array([[1.,1j],[1j,1.]])/np.sqrt(2)
        t=anp.array([[.4,.7]],requires_grad=True)
        def loss(t):
            rho=self.run_program(p,{'a':t,'u':u})
            return anp.real(rho[0,0])+.3*anp.imag(rho[0,1])
        grad=qml.grad(loss)(t);e=1e-6
        for i in range(2):
            d=np.eye(2)[i][None,:]
            self.assertAlmostEqual(float(grad[0,i]),float((loss(t+e*d)-loss(t-e*d))/(2*e)),places=7)

    def test_learning_with_torch_imports_blocked(self):
        import subprocess
        code='''
import sys, importlib.abc
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("PyTorch blocked for this regression")
sys.meta_path.insert(0, BlockTorch())
from examples.train_overlap_embedding import run_demo
report, _ = run_demo(steps=2)
assert not report["torch_imported"]
assert report["max_checkpoint_difference"] < 1e-10
assert report["finite_difference_error"] < 1e-7
assert all(r["finite_gradient"] and r["final_loss"] < r["initial_loss"] for r in report["runs"])
'''
        subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True)

    def test_serialization_types_and_internal_seal(self):
        p=encoded_gram(True); q=Program.from_json(p.to_json())
        self.assertEqual(p.sha256,q.sha256)
        self.assertEqual(q.instructions[-1].result_type,'gram')
        with self.assertRaises(AttributeError): p._name='mutated'
        with self.assertRaises(AttributeError): p._sealed=False
        with self.assertRaises(AttributeError): del p._output
        with self.assertRaises(TypeError):
            Program('bad').emit('a','input',data=np.ones(2))

    def test_unknown_or_modified_tape_and_profile_rejected(self):
        for backend in ('numpy_overlap','pennylane_overlap'):
            with self.assertRaises(UnsupportedLowering):
                self.vm.run(encoded_overlap(), {'left':np.ones(2),'right':np.ones(2)},escort_profile(4),backend)
            record=encoded_overlap().record()
            record['instructions'][2]['attrs']['ignored_gate']=True
            with self.assertRaises(UnsupportedLowering):
                self.run_program(Program.from_json(json.dumps(record)), {'left':np.ones(2),'right':np.ones(2)},backend)
            record=encoded_overlap().record()
            record['instructions'][2]['op']='unknown_encoding'
            with self.assertRaises(UnsupportedLowering):
                self.run_program(Program.from_json(json.dumps(record)), {'left':np.ones(2),'right':np.ones(2)},backend)
            record=encoded_overlap().record()
            record['instructions'][-1]['result_type']='ket'
            with self.assertRaises(UnsupportedLowering):
                self.run_program(Program.from_json(json.dumps(record)), {'left':np.ones(2),'right':np.ones(2)},backend)

    def test_altered_supported_tape_is_executed_not_name_dispatched(self):
        p=encoded_overlap(True)
        record=p.record(); record['return']='left_state'
        early=Program.from_json(json.dumps(record))
        x={'left':np.array([[.4,.6]]),'right':np.array([[1.,-.2]])}
        for backend in ('numpy_overlap','pennylane_overlap'):
            got=self.run_program(early,x,backend)
            expected=np.array([np.cos(.2)*np.exp(-.3j),np.sin(.2)*np.exp(.3j)])
            np.testing.assert_allclose(got,expected,atol=1e-14)

    def test_shape_and_nonfinite_failures(self):
        p=encoded_overlap(True)
        with self.assertRaises(ValueError): self.run_program(p,{'left':np.ones(3),'right':np.ones(3)})
        with self.assertRaises(ValueError): self.run_program(encoded_overlap(),{'left':np.array([np.inf]),'right':np.ones(1)})
        with self.assertRaises(ValueError):
            self.run_program(one_op('inner_product',[('a','ket'),('b','ket')],'complex'),{'a':np.ones(1),'b':normalize(np.ones(2))})

    def test_trace_distinguishes_qnodes_from_host_algebra(self):
        result=self.run_program(encoded_gram(True),{'angles':self.rng.normal(size=(3,2,2))},'pennylane_overlap',audit=True)
        self.assertEqual(result.trace[1]['implementation'],'pennylane_qnode')
        self.assertEqual(result.trace[-1]['implementation'],'host_algebra')
        self.assertEqual(result.numerical_interventions,[])
        self.assertNotIn('torch',sys.modules)


if __name__ == '__main__':
    unittest.main()
