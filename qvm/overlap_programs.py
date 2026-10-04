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


def _gram_program(name, complex_encoding):
    p = Program(name)
    p.input('angles', 'angles')
    encoder = 'encode_ryrz' if complex_encoding else 'encode_ry'
    p.emit('states', encoder, 'angles', role='state', result_type='ket')
    p.emit('collection', 'as_collection', 'states', result_type='state_collection')
    p.emit('gram', 'gram_matrix', 'collection', result_type='gram')
    return p


# ---- I. overlap: what a measurement can see --------------------------------

def overlap_spectrum(complex_encoding=False):
    p = _gram_program('overlap_spectrum', complex_encoding)
    p.emit('spectrum', 'gram_spectrum', 'gram', result_type='spectrum')
    return p.returns('spectrum')


def overlap_participation(complex_encoding=False):
    p = _gram_program('overlap_participation', complex_encoding)
    p.emit('spectrum', 'gram_spectrum', 'gram', result_type='spectrum')
    p.emit('participation', 'spectral_participation', 'spectrum', result_type='real')
    return p.returns('participation')


def overlap_coherence(complex_encoding=False):
    p = _gram_program('overlap_coherence', complex_encoding)
    p.emit('coherence', 'gram_coherence', 'gram', result_type='real')
    return p.returns('coherence')


def phase_ablated_gram(complex_encoding=False):
    p = _gram_program('phase_ablated_gram', complex_encoding)
    p.emit('ablated', 'phase_ablate', 'gram', result_type='gram')
    return p.returns('ablated')


# ---- II. Schrodinger: how overlap is carried --------------------------------

def generator_evolution():
    p = Program('generator_evolution')
    p.input('generator', 'hermitian')
    p.input('time', 'real')
    p.emit('unitary', 'matrix_exponential', 'generator', 'time',
           role='evolution', result_type='unitary')
    return p.returns('unitary')


def generator_spectrum():
    p = Program('generator_spectrum')
    p.input('generator', 'hermitian')
    p.emit('spectrum', 'generator_spectrum', 'generator', result_type='spectrum')
    return p.returns('spectrum')


def commutator_strength():
    p = Program('commutator_strength')
    p.input('left', 'hermitian')
    p.input('right', 'hermitian')
    p.emit('commutator', 'commutator', 'left', 'right', result_type='operator')
    p.emit('strength', 'frobenius_norm', 'commutator', result_type='real')
    return p.returns('strength')


def time_ordered_sequence():
    p = Program('time_ordered_sequence')
    p.input('generators', 'hermitian')
    p.input('times', 'real')
    p.emit('unitary', 'time_ordered_evolve', 'generators', 'times',
           role='evolution', result_type='unitary')
    return p.returns('unitary')


# ---- III. duality: how measurement destroys overlap -------------------------

def detector_duality_program():
    p = Program('detector_duality')
    p.input('left', 'ket')
    p.input('right', 'ket')
    p.emit('duality', 'detector_duality', 'left', 'right', result_type='real')
    return p.returns('duality')


def path_duality_program():
    p = Program('path_duality')
    p.input('state', 'density')
    p.emit('duality', 'path_duality', 'state', result_type='real')
    return p.returns('duality')


def duality_slack_program():
    p = Program('duality_slack')
    p.input('pair', 'real')
    p.emit('slack', 'duality_slack', 'pair', result_type='real')
    return p.returns('slack')


# ---- IV. Schwinger: how a source generates overlap --------------------------

def resolvent_program(truncated=False):
    name = 'proper_time_resolvent' if truncated else 'resolvent'
    p = Program(name)
    p.input('generator', 'hermitian')
    p.input('source_energy', 'real')
    op = 'proper_time_resolvent' if truncated else 'resolvent'
    p.emit('propagator', op, 'generator', 'source_energy', result_type='operator')
    return p.returns('propagator')


def source_response_program():
    p = Program('source_response')
    p.input('propagator', 'operator')
    p.input('source', 'amplitude')
    p.emit('field', 'source_response', 'propagator', 'source', result_type='amplitude')
    return p.returns('field')


def generating_functional_program():
    p = Program('generating_functional')
    p.input('propagator', 'operator')
    p.input('source', 'amplitude')
    p.emit('action', 'generating_functional', 'propagator', 'source', result_type='real')
    return p.returns('action')
