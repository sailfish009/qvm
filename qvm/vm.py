"""Virtual machine.

An immutable Program, a SemanticProfile, and a Backend are independent inputs.
The VM validates inputs, dispatches execution, infers the output contract, and
optionally returns a full audit record.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from .backends.numpy_semantic import NumpySemanticBackend
from .backends.pennylane_standard import PennyLaneStandardBackend
from .backends.overlap import OverlapBackend
from .semantics import standard_profile, SemanticProfile
from .value_types import VALUE_TYPES, validate_value

# Compatibility inference for untyped v2/v3 outputs. Roles are not types.
LEGACY_RESULT_TYPES = {
    'ry_product_state': 'ket', 'pure_fidelity': 'real',
    'density_from_bloch': 'density', 'statevector_density': 'density',
    'unitary': 'density', 'partial_swap': 'density',
    'partial_trace_second': 'density', 'dephase': 'density',
    'kraus_channel': 'density', 'branch_state': 'density', 'tensor': 'density',
}


@dataclass
class ExecutionResult:
    value: object
    program_sha256: str
    program_structural_sha256: str
    semantics: dict
    backend: str
    trace: list
    input_sha256: dict
    used_assumptions: list
    unused_changed_assumptions: list
    numerical_interventions: list

    def audit_record(self):
        return {
            'program_sha256': self.program_sha256,
            'program_structural_sha256': self.program_structural_sha256,
            'semantics': self.semantics,
            'backend': self.backend,
            'trace': self.trace,
            'input_sha256': self.input_sha256,
            'used_assumptions': self.used_assumptions,
            'unused_changed_assumptions': self.unused_changed_assumptions,
            'numerical_interventions': self.numerical_interventions,
        }


class QVM:
    def __init__(self):
        self.backends = {
            'numpy_semantic': NumpySemanticBackend(),
            'pennylane_standard': PennyLaneStandardBackend(),
            'numpy_overlap': OverlapBackend(),
            'pennylane_overlap': OverlapBackend(pennylane=True),
        }

    def register_backend(self, name, backend):
        self.backends[name] = backend

    def _profile(self, semantics):
        if semantics is None or semantics == 'standard':
            return standard_profile()
        if isinstance(semantics, SemanticProfile):
            return semantics
        raise TypeError('semantics must be a SemanticProfile or standard')

    @staticmethod
    def _hash(value):
        array = np.asarray(value)
        header = str((array.dtype.str, array.shape)).encode()
        return hashlib.sha256(header + array.tobytes()).hexdigest()

    def _validate(self, program, inputs, profile):
        if not getattr(program, 'sealed', False):
            raise ValueError('program must be sealed before execution')
        policy = profile.numerical_policy
        for ins in program.instructions:
            if ins.op != 'input':
                continue
            if ins.output not in inputs:
                raise KeyError(f'missing input {ins.output}')
            value = inputs[ins.output]
            shape = ins.attrs.get('shape')
            if hasattr(value, '_value'):
                if shape is not None and tuple(value.shape) != tuple(shape):
                    raise ValueError(f'{ins.output} expected shape {tuple(shape)}')
                continue
            array = np.asarray(value)
            if array.dtype.kind in 'fc' and not np.isfinite(array).all():
                raise ValueError(f'{ins.output} must be finite')
            kind = ins.attrs['qtype']
            if shape is not None and tuple(np.shape(value)) != tuple(shape):
                raise ValueError(f'{ins.output} expected shape {tuple(shape)}')
            if kind in VALUE_TYPES:
                validate_value(value, kind, policy)
            elif kind == 'bloch':
                norms = np.linalg.norm(np.asarray(value, dtype=float), axis=-1)
                if np.any(norms > 1 + policy.bloch_validation_tolerance):
                    raise ValueError('Bloch input outside unit ball for this profile')
            elif kind in ('effects', 'projectors'):
                self._validate_effects(value, kind, policy)
            elif kind == 'kraus':
                self._validate_kraus(value, policy)

    @staticmethod
    def _validate_effects(value, kind, policy):
        effects = np.asarray(value)
        dim = effects.shape[-1]
        if effects.ndim != 3 or effects.shape[-2:] != (dim, dim):
            raise ValueError('invalid effects')
        tolerance = policy.operator_validation_tolerance
        adjoint = effects.conj().transpose(0, 2, 1)
        if not np.allclose(effects, adjoint, atol=tolerance):
            raise ValueError('effects must be Hermitian')
        if min(np.linalg.eigvalsh(effects).reshape(-1)) < -tolerance:
            raise ValueError('effects must be positive semidefinite')
        if not np.allclose(effects.sum(0), np.eye(dim), atol=tolerance):
            raise ValueError('effects must form a complete POVM')
        if kind == 'projectors' and any(not np.allclose(e @ e, e, atol=tolerance) for e in effects):
            raise ValueError('projectors must be idempotent')

    @staticmethod
    def _validate_kraus(value, policy):
        kraus = np.asarray(value)
        total = sum(k.conj().T @ k for k in kraus)
        if not np.allclose(total, np.eye(total.shape[0]),
                           atol=policy.operator_validation_tolerance):
            raise ValueError('Kraus operators are not trace preserving')

    def run(self, program, inputs, semantics=None, backend='numpy_semantic', audit=False):
        profile = self._profile(semantics)
        profile._check()
        if backend not in self.backends:
            raise KeyError(f'unknown backend {backend}')
        self._validate(program, inputs, profile)
        value, trace = self.backends[backend].execute(program, inputs, profile, True)
        output_instruction = next(ins for ins in program.instructions
                                  if ins.output == program.output)
        kind = self._infer_output_type(program, output_instruction, trace, backend)
        if kind:
            validate_value(value, kind, profile.numerical_policy)
        elif output_instruction.role in ('state', 'evolution'):
            profile.validate_state(value)
        if not audit:
            return value
        return self._audit_result(
            program, inputs, profile, backend, value, trace, output_instruction)

    @staticmethod
    def _infer_output_type(program, output_instruction, trace, backend):
        inferred = LEGACY_RESULT_TYPES.get(output_instruction.op)
        if backend in ('numpy_overlap', 'pennylane_overlap'):
            match = next(t for t in trace if t['output'] == program.output)
            inferred = match['result_type']
        if output_instruction.op == 'input' and output_instruction.attrs['qtype'] in VALUE_TYPES:
            inferred = output_instruction.attrs['qtype']
        if inferred and output_instruction.result_type and inferred != output_instruction.result_type:
            raise ValueError('output result_type conflicts with opcode contract')
        return output_instruction.result_type or inferred

    def _audit_result(self, program, inputs, profile, backend, value, trace, output_instruction):
        hashes = {k: self._hash(v) for k, v in inputs.items() if not hasattr(v, '_value')}
        manifest = profile.manifest()
        used_set = {name for item in trace for name in item.get('assumptions_used', [])}
        used_set.add('numerical_policy')
        mapping = {
            'born_exponent': 'measurement_exponent',
            'luders_collapse': 'collapse_strength',
            'linear_evolution': 'evolution_spectral_power',
        }
        unused = [name for name in manifest['changed_assumptions']
                  if mapping[name] not in used_set]
        interventions = [
            {'instruction_index': item.get('index'), 'op': item.get('op'), **event}
            for item in trace for event in item.get('numerical_interventions', [])
        ]
        return ExecutionResult(
            value, program.sha256, program.structural_sha256, manifest, backend,
            trace, hashes, sorted(used_set), unused, interventions)