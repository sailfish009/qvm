"""Small composable programs with no Q/K/V roles or attention backbone."""
from .ir import Program


def encoded_overlap(complex_encoding=False, fidelity=False):
    p = Program('encoded_fidelity' if fidelity else 'encoded_overlap')
    p.input('left', 'angles')
    p.input('right', 'angles')
    op = 'encode_ryrz' if complex_encoding else 'encode_ry'
    p.emit('left_state', op, 'left', role='state', result_type='ket')
    p.emit('right_state', op, 'right', role='state', result_type='ket')
    p.emit('overlap', 'inner_product', 'left_state', 'right_state', result_type='complex')
    if fidelity:
        p.emit('fidelity', 'absolute_square', 'overlap', result_type='real')
        return p.returns('fidelity')
    return p.returns('overlap')


def encoded_gram(complex_encoding=False):
    p = Program('encoded_gram')
    p.input('angles', 'angles')
    p.emit('states', 'encode_ryrz' if complex_encoding else 'encode_ry',
           'angles', role='state', result_type='ket')
    p.emit('collection', 'as_collection', 'states', result_type='state_collection')
    p.emit('gram', 'gram_matrix', 'collection', result_type='gram')
    return p.returns('gram')


def state_superposition(normalized=True):
    p = Program('state_superposition')
    p.input('states', 'state_collection')
    p.input('coefficients', 'coefficients')
    p.emit('amplitude', 'linear_combination', 'states', 'coefficients', result_type='amplitude')
    if normalized:
        p.emit('state', 'normalize', 'amplitude', role='state', result_type='ket')
        return p.returns('state')
    return p.returns('amplitude')


def regularized_span_projection(ridge=1e-3):
    """Ridge smoother, not an exact orthogonal projector or physical gate."""
    p = Program('regularized_span_projection')
    p.input('states', 'state_collection')
    p.input('state', 'ket')
    p.emit('projection', 'ridge_project', 'states', 'state', ridge=float(ridge),
           result_type='amplitude')
    return p.returns('projection')
