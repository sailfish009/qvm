"""Typed, deeply sealed instruction tape.

An instruction records what was computed, not what it physically means.
Roles classify intent; result types are separate value contracts.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import json
import math

from .value_types import VALUE_TYPES

SCHEMA = 'qvm_v0.0005'
KNOWN_SCHEMAS = ('qvm_v0.0002', 'qvm_v0.0003', 'qvm_v0.0004', SCHEMA)
ROLES = ('data', 'state', 'composition', 'evolution', 'measurement', 'host_math', 'output')


def _freeze(value):
    """Copy JSON attributes into an immutable, finite-valued structure."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError('instruction attributes must be finite JSON values')


def _thaw(value):
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class Instruction:
    output: str
    op: str
    inputs: tuple[str, ...] = ()
    role: str = 'host_math'
    attrs: Mapping = field(default_factory=dict)
    result_type: str | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f'unknown semantic role {self.role}')
        if self.result_type is not None and self.result_type not in VALUE_TYPES:
            raise ValueError(f'unknown result type {self.result_type}')
        object.__setattr__(self, 'inputs', tuple(self.inputs))
        object.__setattr__(self, 'attrs', _freeze(self.attrs))

    def record(self):
        record = {
            'output': self.output,
            'op': self.op,
            'inputs': list(self.inputs),
            'role': self.role,
            'attrs': _thaw(self.attrs),
        }
        if self.result_type is not None:
            record['result_type'] = self.result_type
        return record


class Program:
    """Append-only tape that becomes deeply immutable once sealed."""

    def __init__(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError('program name must be nonempty')
        self._name = name
        self._instructions = []
        self._output = None
        self._sealed = False

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('sealed program is immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, '_sealed', False):
            raise AttributeError('sealed program is immutable')
        object.__delattr__(self, name)

    @property
    def name(self):
        return self._name

    @property
    def instructions(self):
        return self._instructions if not self._sealed else self._instructions

    @property
    def output(self):
        return self._output

    @property
    def sealed(self):
        return self._sealed

    def emit(self, output, op, *inputs, role='host_math', result_type=None, **attrs):
        if self._sealed:
            raise RuntimeError('program is sealed')
        known = {ins.output for ins in self._instructions}
        if output in known:
            raise ValueError(f'duplicate value {output}')
        if op != 'input' and any(name not in known for name in inputs):
            raise ValueError('input used before definition')
        instruction = Instruction(output, op, tuple(inputs), role, attrs, result_type)
        self._instructions.append(instruction)
        return output

    def input(self, name, qtype, shape=None):
        return self.emit(name, 'input', role='data', qtype=qtype, shape=shape)

    def returns(self, value):
        if self._sealed:
            raise RuntimeError('program is sealed')
        if value not in {ins.output for ins in self._instructions}:
            raise ValueError('unknown return value')
        self._output = value
        self._instructions = tuple(self._instructions)
        self._sealed = True
        return self

    def record(self):
        if not self._sealed or self._output is None:
            raise ValueError('program is not sealed')
        return {
            'schema': SCHEMA,
            'name': self._name,
            'instructions': [ins.record() for ins in self._instructions],
            'return': self._output,
        }

    def structural_record(self):
        record = self.record()
        return {'instructions': record['instructions'], 'return': record['return']}

    def to_json(self, indent=None):
        separators = (',', ':') if indent is None else None
        return json.dumps(self.record(), sort_keys=True, separators=separators, indent=indent)

    @property
    def sha256(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @property
    def structural_sha256(self):
        payload = json.dumps(self.structural_record(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_json(cls, text):
        record = json.loads(text)
        if record.get('schema') not in KNOWN_SCHEMAS:
            raise ValueError('wrong schema')
        program = cls(record['name'])
        for item in record['instructions']:
            program.emit(
                item['output'], item['op'], *item['inputs'],
                role=item['role'], result_type=item.get('result_type'), **item['attrs'])
        return program.returns(record['return'])