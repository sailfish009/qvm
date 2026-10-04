"""Array value contracts, independent of instruction roles and physical claims.

Batch axes are leading axes. Validation uses primal values during Autograd
tracing; it never inserts a projection, detaches the expression, or repairs data.
"""
import numpy as np

VALUE_TYPES = frozenset({
    'ket', 'amplitude', 'state_collection', 'density', 'gram', 'operator',
    'hermitian', 'unitary', 'complex', 'real', 'angles', 'coefficients',
    'spectrum',
})

VECTORS = frozenset({'ket', 'amplitude', 'state_collection'})
MATRICES = frozenset({'density', 'gram', 'operator', 'hermitian', 'unitary'})


def primal(value):
    while hasattr(value, '_value'):
        value = value._value
    return np.asarray(value)


def validate_value(value, kind, policy):
    if kind not in VALUE_TYPES:
        raise ValueError(f'unknown value type: {kind}')
    array = primal(value)
    if array.dtype.kind not in 'biufc' or not np.isfinite(array).all():
        raise ValueError(f'{kind}: expected finite numeric values')
    if array.size == 0:
        raise ValueError(f'{kind}: empty values are not supported')
    tolerance = policy.state_validation_tolerance

    if kind in ('real', 'angles', 'spectrum') and np.iscomplexobj(array):
        raise ValueError(f'{kind}: expected real dtype')

    if kind in ('ket', 'amplitude', 'state_collection', 'coefficients', 'spectrum'):
        if array.ndim < (2 if kind == 'state_collection' else 1):
            raise ValueError(f'{kind}: missing vector/collection axes')

    if kind in ('ket', 'state_collection'):
        norms = np.sum(np.abs(array) ** 2, axis=-1)
        if not np.allclose(norms, 1, atol=tolerance, rtol=0):
            raise ValueError(f'{kind}: vectors must have norm one')

    if kind in MATRICES:
        if array.ndim < 2 or array.shape[-1] != array.shape[-2]:
            raise ValueError(f'{kind}: expected square last two axes')
        adjoint = np.swapaxes(array.conj(), -1, -2)
        if kind in ('density', 'gram', 'hermitian'):
            if not np.allclose(array, adjoint, atol=tolerance, rtol=0):
                raise ValueError(f'{kind}: expected Hermitian matrix')
        if kind in ('density', 'gram'):
            if np.min(np.linalg.eigvalsh(array)) < -tolerance:
                raise ValueError(f'{kind}: expected positive semidefinite matrix')
        if kind == 'density':
            if not np.allclose(np.trace(array, axis1=-2, axis2=-1), 1, atol=tolerance, rtol=0):
                raise ValueError('density: trace must be one')
        if kind == 'unitary':
            eye = np.eye(array.shape[-1])
            if not np.allclose(adjoint @ array, eye,
                               atol=policy.operator_validation_tolerance, rtol=0):
                raise ValueError('unitary: expected U†U = I')
    return value