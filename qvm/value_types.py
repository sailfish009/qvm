"""Array value contracts, independent of instruction roles and physical claims.

Batch axes are leading axes. Validation uses primal values during Autograd tracing;
this does not insert a projection, detach the executed expression, or repair data.
"""
import numpy as np

VALUE_TYPES = frozenset({
    'ket', 'amplitude', 'state_collection', 'density', 'gram', 'operator',
    'hermitian', 'unitary', 'complex', 'real', 'angles', 'coefficients',
})


def primal(value):
    while hasattr(value, '_value'):
        value = value._value
    return np.asarray(value)


def validate_value(value, kind, policy):
    if kind not in VALUE_TYPES:
        raise ValueError(f'unknown value type: {kind}')
    a = primal(value)
    if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():
        raise ValueError(f'{kind}: expected finite numeric values')
    if a.size == 0:
        raise ValueError(f'{kind}: empty values are not supported')
    tol = policy.state_validation_tolerance
    if kind in ('real', 'angles'):
        if np.iscomplexobj(a):
            raise ValueError(f'{kind}: expected real dtype')
    if kind in ('ket', 'amplitude', 'state_collection', 'coefficients'):
        if a.ndim < (2 if kind == 'state_collection' else 1):
            raise ValueError(f'{kind}: missing vector/collection axes')
    if kind in ('ket', 'state_collection'):
        norms = np.sum(np.abs(a)**2, axis=-1)
        if not np.allclose(norms, 1, atol=tol, rtol=0):
            raise ValueError(f'{kind}: vectors must have norm one')
    if kind in ('density', 'gram', 'operator', 'hermitian', 'unitary'):
        if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
            raise ValueError(f'{kind}: expected square last two axes')
        adjoint = np.swapaxes(a.conj(), -1, -2)
        if kind in ('density', 'gram', 'hermitian'):
            if not np.allclose(a, adjoint, atol=tol, rtol=0):
                raise ValueError(f'{kind}: expected Hermitian matrix')
        if kind in ('density', 'gram'):
            if np.min(np.linalg.eigvalsh(a)) < -tol:
                raise ValueError(f'{kind}: expected positive semidefinite matrix')
        if kind == 'density':
            if not np.allclose(np.trace(a, axis1=-2, axis2=-1), 1, atol=tol, rtol=0):
                raise ValueError('density: trace must be one')
        if kind == 'unitary':
            if not np.allclose(adjoint @ a, np.eye(a.shape[-1]),
                               atol=policy.operator_validation_tolerance, rtol=0):
                raise ValueError('unitary: expected U†U = I')
    return value
