"""Composable standard-complex overlap dialect, without attention or QKV.

Both backends execute the same validated tape. The PennyLane backend independently
prepares states using differentiable QNodes; subsequent algebra is explicit host
math shared with the NumPy executor, NOT independent circuit verification of all ops.
"""
from functools import lru_cache
import numpy as np
import pennylane as qml
from pennylane import numpy as anp
from ..numerics import NumericalDomainError
from ..value_types import VALUE_TYPES, primal, validate_value
from .pennylane_standard import UnsupportedLowering

# opcode: (input contracts, output contract, required attribute names)
VECTORS = frozenset({'ket', 'amplitude', 'state_collection'})
SPEC = {
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


def execute_op(ins, args, pennylane, interventions):
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
                ins, [env[x] for x in ins.inputs], self.pennylane, interventions)
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
