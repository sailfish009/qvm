"""Composable standard-complex overlap dialect, without attention or QKV.

Both backends execute the same validated tape. The PennyLane backend independently
prepares states using differentiable QNodes; subsequent algebra is explicit host
math shared with the NumPy executor, NOT independent circuit verification of all ops.
"""
from functools import lru_cache
import numpy as np
import pennylane as qml
from pennylane import numpy as anp
from ..numerics import NumericalDomainError, ordinary
from ..value_types import VALUE_TYPES, primal, validate_value
from .pennylane_standard import UnsupportedLowering

# opcode: (input contracts, output contract, required attribute names)
VECTORS = frozenset({'ket', 'amplitude', 'state_collection'})
OPERATORS = frozenset({'operator', 'hermitian', 'unitary'})
SPEC = {
    # I. overlap: what a measurement can see
    'encode_ry': (('angles',), 'ket', frozenset()),
    'encode_ryrz': (('angles',), 'ket', frozenset()),
    'inner_product': ((VECTORS, VECTORS), 'complex', frozenset()),
    'gram_matrix': (('state_collection',), 'gram', frozenset()),
    'absolute_square': ((frozenset({'complex', 'gram'}),), 'real', frozenset()),
    'linear_combination': (('state_collection', 'coefficients'), 'amplitude', frozenset()),
    'normalize': (('amplitude',), 'ket', frozenset()),
    'as_collection': (('ket',), 'state_collection', frozenset()),
    'ket_density': (('ket',), 'density', frozenset()),
    'apply_unitary': (('ket', 'unitary'), 'ket', frozenset()),
    'expectation': (('ket', 'hermitian'), 'real', frozenset()),
    'cyclic_overlap': (('gram',), 'complex', frozenset({'indices'})),
    'ridge_project': (('state_collection', 'ket'), 'amplitude', frozenset({'ridge'})),
    'gram_spectrum': (('gram',), 'spectrum', frozenset()),
    'spectral_participation': (('spectrum',), 'real', frozenset()),
    'gram_coherence': (('gram',), 'real', frozenset()),
    'phase_ablate': (('gram',), 'gram', frozenset()),
    # II. Schrodinger: how overlap is carried by a generator
    'generator_spectrum': (('hermitian',), 'spectrum', frozenset()),
    'matrix_exponential': (('hermitian', 'real'), 'unitary', frozenset()),
    'time_ordered_evolve': (('hermitian', 'real'), 'unitary', frozenset()),
    'commutator': (('hermitian', 'hermitian'), 'operator', frozenset()),
    'frobenius_norm': ((OPERATORS,), 'real', frozenset()),
    # III. duality: how measurement destroys overlap
    'detector_duality': (('ket', 'ket'), 'real', frozenset()),
    'path_duality': (('density',), 'real', frozenset()),
    'duality_slack': (('real',), 'real', frozenset()),
    # IV. Schwinger: how a source generates overlap
    'resolvent': (('hermitian', 'real'), 'operator', frozenset()),
    'proper_time_resolvent': (('hermitian', 'real'), 'operator', frozenset()),
    'source_response': ((OPERATORS, 'amplitude'), 'amplitude', frozenset()),
    'generating_functional': ((OPERATORS, 'amplitude'), 'real', frozenset()),
}


def check_tape(program):
    """Fail before execution on unknown instructions, types, or ignored attributes."""
    types = {}
    for ins in program.instructions:
        if ins.op == 'input':
            if ins.inputs or set(ins.attrs) != {'qtype', 'shape'}:
                raise UnsupportedLowering('invalid overlap input declaration')
            kind = ins.attrs['qtype']
            if kind not in VALUE_TYPES:
                raise UnsupportedLowering(f'unsupported overlap input type: {kind}')
        else:
            if ins.op not in SPEC:
                raise UnsupportedLowering(f'unsupported overlap opcode: {ins.op}')
            contracts, kind, attrs = SPEC[ins.op]
            if set(ins.attrs) != attrs or len(ins.inputs) != len(contracts):
                raise UnsupportedLowering(f'{ins.op}: unsupported attributes or arity')
            for name, contract in zip(ins.inputs, contracts):
                allowed = {contract} if isinstance(contract, str) else contract
                if types[name] not in allowed:
                    raise UnsupportedLowering(f'{ins.op}: {name} has type {types[name]}')
            if ins.op == 'cyclic_overlap':
                ids = ins.attrs['indices']
                if not isinstance(ids, tuple) or len(ids) < 3 or any(type(i) is not int or i < 0 for i in ids):
                    raise UnsupportedLowering('cyclic_overlap requires at least three nonnegative indices')
            if ins.op == 'ridge_project':
                ridge = ins.attrs['ridge']
                if type(ridge) not in (int, float) or not np.isfinite(ridge) or ridge <= 0:
                    raise UnsupportedLowering('ridge_project requires an explicit finite ridge > 0')
        if ins.result_type is not None and ins.result_type != kind:
            raise UnsupportedLowering(f'{ins.op}: result_type disagrees with operator contract')
        types[ins.output] = kind
    return types


def _encode_numpy(angles, complex_encoding):
    theta = angles[..., 0] if complex_encoding else angles
    state = anp.ones(theta.shape[:-1] + (1,), dtype=complex)
    for i in range(theta.shape[-1]):
        c, s = anp.cos(theta[..., i] / 2), anp.sin(theta[..., i] / 2)
        if complex_encoding:
            phase = angles[..., i, 1]
            c, s = c * anp.exp(-.5j * phase), s * anp.exp(.5j * phase)
        factor = anp.stack((c, s), axis=-1)
        joint = state[..., :, None] * factor[..., None, :]
        state = anp.reshape(joint, theta.shape[:-1] + (-1,))
    return state


@lru_cache(maxsize=32)
def _state_qnode(n, complex_encoding):
    device = qml.device('default.qubit', wires=n, shots=None)

    @qml.qnode(device, interface='autograd', diff_method='backprop')
    def state(angles):
        for wire in range(n):
            if complex_encoding:
                qml.RY(angles[wire, 0], wires=wire)
                qml.RZ(angles[wire, 1], wires=wire)
            else:
                qml.RY(angles[wire], wires=wire)
        return qml.state()
    return state


def encode(angles, complex_encoding=False, pennylane=False):
    shape = angles.shape
    if complex_encoding:
        if len(shape) < 2 or shape[-1] != 2 or shape[-2] < 1:
            raise ValueError('encode_ryrz expects (..., wires, 2)')
        n, leading, tail = shape[-2], shape[:-2], shape[-2:]
    else:
        if len(shape) < 1 or shape[-1] < 1:
            raise ValueError('encode_ry expects (..., wires)')
        n, leading, tail = shape[-1], shape[:-1], shape[-1:]
    if not pennylane:
        return _encode_numpy(angles, complex_encoding)
    rows = anp.reshape(angles, (-1,) + tail)
    # Individual executions avoid wide-state PennyLane broadcasting limitations.
    values = anp.stack([_state_qnode(n, complex_encoding)(row) for row in rows])
    return anp.reshape(values, leading + (2**n,))


def gram(states):
    return anp.einsum('...id,...jd->...ij', anp.conj(states), states)


def normalize(amplitude):
    # Rescale first, so representable tiny/huge nonzero amplitudes remain valid.
    # Component-wise max also avoids overflow of abs(x+iy) near float64 max.
    scale = anp.maximum(anp.max(anp.abs(anp.real(amplitude)), axis=-1, keepdims=True),
                        anp.max(anp.abs(anp.imag(amplitude)), axis=-1, keepdims=True))
    if np.any(primal(scale) == 0):
        raise NumericalDomainError('normalize: zero amplitude has no normalized state')
    scaled = amplitude / scale
    return scaled / anp.sqrt(anp.sum(anp.real(anp.conj(scaled) * scaled), axis=-1, keepdims=True))


def _unitary_from_hermitian(operator, times):
    eigenvalues, eigenvectors = anp.linalg.eigh(operator)
    phases = anp.exp(-1j * anp.asarray(times)[..., None] * eigenvalues)
    return (eigenvectors * phases) @ anp.conj(eigenvectors.swapaxes(-1, -2))


def _time_ordered(operators, times):
    """Ordered product U = P_{K-1} ... P_0 with P_k = exp(-i t_k H_k)."""
    count = operators.shape[-3]
    dim = operators.shape[-1]
    result = anp.eye(dim, dtype=complex)
    for k in range(count):
        result = _unitary_from_hermitian(operators[..., k, :, :], times[..., k]) @ result
    return result


def _resolvent_kernel(operator, omega, policy, truncated):
    eigenvalues, eigenvectors = anp.linalg.eigh(operator)
    shifted = eigenvalues - omega
    if ordinary(shifted) and float(anp.min(shifted)) <= 0:
        raise NumericalDomainError('resolvent: shift must keep the spectrum positive')
    if truncated:
        weights = (1 - anp.exp(-shifted * policy.proper_time_cutoff)) / shifted
    else:
        weights = 1 / shifted
    return (eigenvectors * weights) @ anp.conj(eigenvectors.swapaxes(-1, -2))


def _detector_duality(left, right):
    overlap = anp.sum(anp.conj(left) * right, axis=-1)
    visibility = anp.abs(overlap)
    distinguishability = anp.sqrt(anp.maximum(1 - visibility ** 2, 0))
    return anp.stack((distinguishability, visibility), axis=-1)


def execute_op(ins, args, pennylane, interventions, policy):
    op = ins.op
    if op in ('encode_ry', 'encode_ryrz'):
        return encode(args[0], op == 'encode_ryrz', pennylane)
    if op == 'inner_product':
        if args[0].shape[-1] != args[1].shape[-1]:
            raise ValueError('inner_product: Hilbert dimensions differ')
        return anp.sum(anp.conj(args[0]) * args[1], axis=-1)
    if op == 'gram_matrix':
        return gram(args[0])
    if op == 'absolute_square':
        return anp.real(anp.conj(args[0]) * args[0])
    if op == 'as_collection':
        if args[0].ndim < 2:
            raise ValueError('as_collection requires a collection axis')
        return args[0]
    if op == 'linear_combination':
        states, coefficients = args
        if states.shape[-2] != coefficients.shape[-1]:
            raise ValueError('linear_combination: coefficient count differs')
        return anp.einsum('...n,...nd->...d', coefficients, states)
    if op == 'normalize':
        return normalize(args[0])
    if op == 'ket_density':
        return args[0][..., :, None] * anp.conj(args[0][..., None, :])
    if op in ('apply_unitary', 'expectation'):
        state, operator = args
        if state.shape[-1] != operator.shape[-1]:
            raise ValueError(f'{op}: Hilbert dimensions differ')
        transformed = anp.einsum('...de,...e->...d', operator, state)
        return transformed if op == 'apply_unitary' else anp.real(anp.sum(anp.conj(state) * transformed, axis=-1))
    if op == 'cyclic_overlap':
        matrix = args[0]
        indices = ins.attrs['indices']
        if max(indices) >= matrix.shape[-1]:
            raise ValueError('cyclic_overlap: index outside Gram matrix')
        value = 1. + 0j
        for i, j in zip(indices, indices[1:] + indices[:1]):
            value = value * matrix[..., i, j]
        return value
    if op == 'ridge_project':
        states, state = args
        if states.shape[-1] != state.shape[-1]:
            raise ValueError('ridge_project: Hilbert dimensions differ')
        ridge = ins.attrs['ridge']
        matrix = gram(states) + ridge * anp.eye(states.shape[-2])
        z = anp.einsum('...nd,...d->...n', anp.conj(states), state)
        # NumPy >=2 requires explicit RHS column for batched vector solve.
        coefficients = anp.linalg.solve(matrix, z[..., None])[..., 0]
        interventions.append({'kind': 'declared_ridge_regularization', 'ridge': ridge,
                              'operation': 'ridge_project', 'exact_projector': False})
        return anp.einsum('...n,...nd->...d', coefficients, states)
    if op == 'gram_spectrum':
        return anp.linalg.eigvalsh(args[0])
    if op == 'generator_spectrum':
        return anp.linalg.eigvalsh(args[0])
    if op == 'spectral_participation':
        spectrum = args[0]
        total = anp.sum(spectrum, axis=-1)
        squared = anp.sum(anp.real(spectrum) ** 2, axis=-1)
        if ordinary(squared) and float(anp.min(squared)) <= 0:
            raise NumericalDomainError('spectral_participation: zero spectrum has no participation')
        return total ** 2 / squared
    if op == 'gram_coherence':
        matrix = args[0]
        energy = anp.sum(anp.real(anp.conj(matrix) * matrix), axis=(-2, -1))
        diagonal = anp.diagonal(matrix, axis1=-2, axis2=-1)
        diag_energy = anp.sum(anp.real(anp.conj(diagonal) * diagonal), axis=-1)
        return anp.sqrt(anp.maximum(energy - diag_energy, 0))
    if op == 'phase_ablate':
        matrix = args[0]
        diagonal = anp.diagonal(matrix, axis1=-2, axis2=-1)
        return anp.eye(matrix.shape[-1]) * diagonal[..., :, None]
    if op == 'matrix_exponential':
        return _unitary_from_hermitian(args[0], args[1])
    if op == 'time_ordered_evolve':
        return _time_ordered(args[0], args[1])
    if op == 'commutator':
        return args[0] @ args[1] - args[1] @ args[0]
    if op == 'frobenius_norm':
        return anp.sqrt(anp.real(anp.sum(anp.conj(args[0]) * args[0], axis=(-2, -1))))
    if op == 'detector_duality':
        return _detector_duality(args[0], args[1])
    if op == 'path_duality':
        density = args[0]
        if density.shape[-1] != 2:
            raise ValueError('path_duality requires a qubit density matrix')
        predictability = anp.abs(density[..., 0, 0] - density[..., 1, 1])
        visibility = 2 * anp.abs(density[..., 0, 1])
        return anp.stack((predictability, visibility), axis=-1)
    if op == 'duality_slack':
        pair = args[0]
        return 1 - pair[..., 0] ** 2 - pair[..., 1] ** 2
    if op == 'resolvent':
        return _resolvent_kernel(args[0], args[1], policy, truncated=False)
    if op == 'proper_time_resolvent':
        return _resolvent_kernel(args[0], args[1], policy, truncated=True)
    if op == 'source_response':
        return anp.einsum('...de,...e->...d', args[0], args[1])
    if op == 'generating_functional':
        applied = anp.einsum('...de,...e->...d', args[0], args[1])
        return 0.5 * anp.real(anp.sum(anp.conj(args[1]) * applied, axis=-1))
    raise UnsupportedLowering(op)


class OverlapBackend:
    def __init__(self, pennylane=False):
        self.pennylane = pennylane
        self.name = 'pennylane_overlap' if pennylane else 'numpy_overlap'

    def execute(self, program, inputs, semantics, return_trace=False):
        # This dialect is standard complex Hilbert algebra, not a counterfactual lowering.
        if semantics.manifest()['changed_assumptions']:
            raise UnsupportedLowering(f'{self.name} supports standard semantics only')
        types = check_tape(program)
        env, trace = {}, []
        for index, ins in enumerate(program.instructions):
            interventions = []
            value = inputs[ins.output] if ins.op == 'input' else execute_op(
                ins, [env[x] for x in ins.inputs], self.pennylane, interventions,
                semantics.numerical_policy)
            validate_value(value, types[ins.output], semantics.numerical_policy)
            env[ins.output] = value
            prepares = ins.op in ('encode_ry', 'encode_ryrz')
            used = ['state_space'] if ins.op != 'input' else []
            if prepares:
                used.append('composition_rule')
            if ins.op == 'apply_unitary':
                used.extend(['evolution_spectral_power', 'evolution_application_scope'])
            trace.append({'index': index, 'output': ins.output, 'op': ins.op,
                          'role': ins.role, 'result_type': types[ins.output],
                          'assumptions_used': used, 'numerical_interventions': interventions,
                          'implementation': ('pennylane_qnode' if self.pennylane else 'numpy_product_state')
                          if prepares else 'host_algebra'})
        result = env[program.output]
        return (result, trace) if return_trace else result
