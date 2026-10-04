"""qvm_v0.0005: auditable quantum-equation execution.

Four concept layers share one opt-in typed dialect:
  I   overlap     (observable)      overlap_spectrum, overlap_participation, ...
  II  Schrodinger (dynamics)        generator_evolution, time_ordered_sequence, ...
  III duality     (constraint)      detector_duality_program, path_duality_program
  IV  Schwinger   (response)        resolvent_program, generating_functional_program
Legacy counterfactual semantics and both NumPy/PennyLane lowerings are preserved.
"""
from .ir import Program, Instruction
from .vm import QVM, ExecutionResult
from .numerics import NumericalPolicy, NumericalDomainError, UndefinedBranch
from .semantics import (
    SemanticProfile, SelectiveMeasurementResult, standard_profile, escort_profile,
    partial_collapse_profile, spectral_power_profile, counterfactual_profile,
)
from .programs import (
    product_ry_fidelity, povm_measurement, projective_measurement, unitary_evolution,
    partial_swap, mixed_bures, sandwiched_renyi, kraus_channel,
)
from .backends.pennylane_standard import UnsupportedLowering
from .properties import PropertyResult, inspect_profile, default_property_report
from .overlap_programs import (
    encoded_overlap, encoded_gram, state_superposition, regularized_span_projection,
    overlap_spectrum, overlap_participation, overlap_coherence, phase_ablated_gram,
    generator_evolution, generator_spectrum, commutator_strength, time_ordered_sequence,
    detector_duality_program, path_duality_program, duality_slack_program,
    resolvent_program, source_response_program, generating_functional_program,
)
from .value_types import validate_value, VALUE_TYPES

__version__ = '0.0.5'

__all__ = [
    'PropertyResult', 'inspect_profile', 'default_property_report',
    'Program', 'Instruction', 'QVM', 'ExecutionResult',
    'NumericalPolicy', 'NumericalDomainError', 'UndefinedBranch',
    'SemanticProfile', 'SelectiveMeasurementResult', 'standard_profile',
    'escort_profile', 'partial_collapse_profile', 'spectral_power_profile',
    'counterfactual_profile',
    'product_ry_fidelity', 'povm_measurement', 'projective_measurement',
    'unitary_evolution', 'partial_swap', 'mixed_bures', 'sandwiched_renyi',
    'kraus_channel', 'UnsupportedLowering',
    'encoded_overlap', 'encoded_gram', 'state_superposition',
    'regularized_span_projection',
    'overlap_spectrum', 'overlap_participation', 'overlap_coherence',
    'phase_ablated_gram', 'generator_evolution', 'generator_spectrum',
    'commutator_strength', 'time_ordered_sequence', 'detector_duality_program',
    'path_duality_program', 'duality_slack_program', 'resolvent_program',
    'source_response_program', 'generating_functional_program',
    'validate_value', 'VALUE_TYPES',
]