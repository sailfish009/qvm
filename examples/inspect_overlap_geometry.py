"""No-QKV overlap diagnostics: gauge, superposition, Gram and regularized span.

These are equation checks, not learning-performance gates or physical-law tests.
"""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvm import (QVM, Program, encoded_overlap, encoded_gram,
                 state_superposition, regularized_span_projection, __version__)


def main():
    rng, vm = np.random.default_rng(4004), QVM()
    p = encoded_overlap(complex_encoding=True)
    errors = []
    for _ in range(100):
        x = {'left': rng.normal(size=(3,2,2)), 'right': rng.normal(size=(3,2,2))}
        a = vm.run(p, x, backend='numpy_overlap')
        b = vm.run(p, x, backend='pennylane_overlap')
        errors.append(float(np.max(abs(a-b))))
    states = rng.normal(size=(4,3)) + 1j*rng.normal(size=(4,3))
    states /= np.linalg.norm(states,axis=-1,keepdims=True)
    coefficients = rng.normal(size=4)+1j*rng.normal(size=4)
    gram = states.conj()@states.T
    u = vm.run(state_superposition(False), {'states':states,'coefficients':coefficients}, backend='numpy_overlap')
    norm = float(np.vdot(u,u).real)
    cgc = float(np.vdot(coefficients,gram@coefficients).real)
    phase = np.exp(1j*rng.normal(size=4))
    rephased = states*phase[:,None]
    changed_gram = rephased.conj()@rephased.T
    cycle = lambda g: g[0,1]*g[1,2]*g[2,0]
    state = states[0]
    proj = regularized_span_projection(.05)
    original = vm.run(proj, {'states':states,'state':state}, backend='numpy_overlap',audit=True)
    changed = vm.run(proj, {'states':rephased,'state':state}, backend='numpy_overlap')
    # Relative coefficient phase changes a coherent sum, but not coefficient magnitudes.
    two = np.array([[1.,0.],[0.,1.]])
    plus = vm.run(state_superposition(), {'states':two,'coefficients':np.array([1.,1.])}, backend='numpy_overlap')
    minus = vm.run(state_superposition(), {'states':two,'coefficients':np.array([1.,-1.])}, backend='numpy_overlap')
    observable=np.array([[0.,1.],[1.,0.]])
    report = {'qvm_version':__version__, 'random_cases':100,
              'max_numpy_pennylane_overlap_error':max(errors),
              'superposition_norm_identity_error':abs(norm-cgc),
              'cyclic_overlap_rephasing_error':float(abs(cycle(gram)-cycle(changed_gram))),
              'ridge_projection_rephasing_error':float(np.max(abs(original.value-changed))),
              'gram_eigenvalues':np.linalg.eigvalsh(gram).tolist(),
              'fixed_reference_relative_phase_readouts':[float(np.vdot(s,observable@s).real) for s in (plus,minus)],
              'projection_audit':original.audit_record(),
              'interpretation':'Standard-complex algebra checks, no attention/QKV requirement. Sum norm is NOT a physical success probability without a specified contractive implementation.'}
    assert max(errors)<2e-14 and abs(norm-cgc)<1e-12
    assert report['cyclic_overlap_rephasing_error']<1e-12
    assert report['ridge_projection_rephasing_error']<1e-12
    out=Path(__file__).resolve().parents[1]/'artifacts';out.mkdir(exist_ok=True)
    (out/'overlap_geometry.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
