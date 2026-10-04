"""Replaceable semantic assumptions with an explicit, separate numerical policy.

Semantics answer "what is the physical law"; the numerical policy answers
"how do we handle numerically singular or borderline cases". They are never
conflated.
"""
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
from pennylane import numpy as anp

from .numerics import NumericalPolicy, NumericalDomainError, ordinary, scalar_value


@dataclass(frozen=True)
class SelectiveMeasurementResult:
    probabilities: object
    branches: object
    defined: object
    born_probabilities: object

    def __iter__(self):
        return iter((self.probabilities, self.branches))

    def __getitem__(self, index):
        return (self.probabilities, self.branches, self.defined, self.born_probabilities)[index]


@dataclass(frozen=True)
class SemanticProfile:
    name: str
    measurement_exponent: object = 2.0
    collapse_strength: object = 1.0
    evolution_spectral_power: object = 1.0
    state_space: str = 'complex_density'
    composition_rule: str = 'tensor_product'
    mixture_rule: str = 'convex_linear'
    numerical_policy: NumericalPolicy = field(default_factory=NumericalPolicy)
    evolution_application_scope: str = 'per_evolution_instruction'

    def _check(self):
        if self.state_space != 'complex_density':
            raise NotImplementedError('alternative state spaces are not implemented in v0.0005')
        if self.composition_rule != 'tensor_product':
            raise NotImplementedError('alternative composition rules are not implemented in v0.0005')
        if self.mixture_rule != 'convex_linear':
            raise NotImplementedError('alternative input-mixture rules are not implemented in v0.0005')
        if self.evolution_application_scope != 'per_evolution_instruction':
            raise NotImplementedError('alternative evolution event scopes are not implemented in v0.0005')
        for key, value in (('measurement_exponent', self.measurement_exponent),
                           ('collapse_strength', self.collapse_strength),
                           ('evolution_spectral_power', self.evolution_spectral_power)):
            if ordinary(value):
                number = scalar_value(value)
                if not np.isfinite(number):
                    raise ValueError(f'{key} must be finite')
                if key == 'collapse_strength':
                    if not 0 <= number <= 1:
                        raise ValueError(f'invalid {key}')
                elif number <= 0:
                    raise ValueError(f'invalid {key}')

    def manifest(self):
        self._check()
        alpha = scalar_value(self.measurement_exponent)
        kappa = scalar_value(self.collapse_strength)
        beta = scalar_value(self.evolution_spectral_power)
        changed = []
        if alpha != 2:
            changed.append('born_exponent')
        if kappa != 1:
            changed.append('luders_collapse')
        if beta != 1:
            changed.append('linear_evolution')
        record = {
            'schema': 'qvm-semantics-v2',
            'name': self.name,
            'state_space': self.state_space,
            'composition': self.composition_rule,
            'evolution': 'linear_cptp' if beta == 1 else 'spectral_power_after_map',
            'evolution_application_scope': self.evolution_application_scope,
            'measurement': 'born' if alpha == 2 else 'escort_born',
            'measurement_exponent': alpha,
            'collapse': 'luders' if kappa == 1 else ('none' if kappa == 0 else 'partial_luders_mix'),
            'collapse_strength': kappa,
            'mixture': self.mixture_rule,
            'evolution_spectral_power': beta,
            'changed_assumptions': changed,
            'enforced_invariants': ['hermitian', 'positive_semidefinite', 'trace_one'],
            'known_nonstandard_properties': (
                [] if not changed else self._violations(alpha, kappa, beta)),
            'numerical_policy': self.numerical_policy.manifest(),
            'uncovered_alternatives': [
                'alternative_state_spaces', 'alternative_scalar_fields',
                'alternative_composition_rules', 'nonconvex_input_mixtures',
                'unknown_alternatives',
            ],
        }
        record['sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        return record

    @staticmethod
    def _violations(alpha, kappa, beta):
        violations = []
        if alpha != 2:
            violations += [
                'standard_born_rule',
                'standard_noncontextual_measurement_assignment_not_assumed',
                'outcome_refinement_invariance_not_guaranteed',
            ]
        if kappa != 1:
            violations += [
                'standard_selective_measurement_update',
                'repeatability_not_guaranteed',
            ]
        if beta != 1:
            violations += [
                'linearity_of_state_evolution',
                'convex_mixture_preservation_not_guaranteed',
                'instruction_decomposition_invariance_not_guaranteed',
            ]
        return violations

    def validate_state(self, rho, tol=None):
        if not ordinary(rho):
            return
        tolerance = self.numerical_policy.state_validation_tolerance if tol is None else tol
        array = np.asarray(rho)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError('state must be square')
        if not np.isfinite(array).all():
            raise ValueError('state must be finite')
        if not np.allclose(array, array.conj().T, atol=tolerance):
            raise ValueError('state must be Hermitian')
        if not np.isclose(np.trace(array), 1, atol=tolerance):
            raise ValueError('state must have trace one')
        if np.linalg.eigvalsh(array).min() < -tolerance:
            raise ValueError('state must be positive semidefinite')

    def compose(self, a, b, audit=None):
        return anp.kron(a, b)

    def after_evolution(self, rho, audit=None):
        beta = self.evolution_spectral_power
        if ordinary(beta) and scalar_value(beta) == 1:
            return rho
        out = self.numerical_policy.power(rho, beta, audit, 'post_evolution_positive_power')
        return out / anp.real(anp.trace(out))

    def unitary(self, rho, u, audit=None):
        return self.after_evolution(u @ rho @ anp.conj(u.T), audit)

    def channel(self, rho, kraus, audit=None):
        return self.after_evolution(sum(k @ rho @ anp.conj(k.T) for k in kraus), audit)

    def probabilities(self, rho, effects, audit=None):
        raw = anp.stack(tuple(anp.real(anp.trace(rho @ e)) for e in effects))
        if ordinary(raw):
            values = np.asarray(raw, dtype=float)
            if not np.isfinite(values).all():
                raise NumericalDomainError('measurement probabilities are non-finite')
            minimum = float(values.min())
            if minimum < -self.numerical_policy.probability_roundoff_tolerance:
                raise NumericalDomainError(
                    'measurement produced a negative probability beyond roundoff tolerance')
            count = int(np.sum(values < 0))
            if count and audit is not None:
                audit.append({
                    'kind': 'negative_probability_roundoff_projection',
                    'count': count,
                    'minimum': minimum,
                    'replacement': 0.0,
                })
        raw = anp.maximum(raw, 0)
        scale = anp.max(raw)
        if ordinary(scale) and scalar_value(scale) <= 0:
            raise NumericalDomainError('measurement has no positive outcome weight')
        scaled = raw / scale
        power = self.measurement_exponent / 2
        weights = scaled ** power
        total = anp.sum(weights)
        if ordinary(total):
            value = scalar_value(total)
            if not np.isfinite(value) or value <= 0:
                raise NumericalDomainError('measurement normalization failed')
            if audit is not None:
                underflow = int(np.sum((np.asarray(raw) > 0) & (np.asarray(weights) == 0)))
                if underflow:
                    audit.append({
                        'kind': 'positive_measurement_weight_underflow',
                        'count': underflow,
                        'normalization': 'max_rescaled_power',
                    })
        return weights / total

    def selective_projective_measurement(self, rho, projectors, audit=None):
        probabilities = self.probabilities(rho, projectors, audit)
        branches = []
        defined = []
        born_values = []
        kappa = self.collapse_strength
        for projector in projectors:
            born = anp.real(anp.trace(rho @ projector))
            born_values.append(born)
            valid = born > 0
            safe = anp.where(valid, born, 1.0)
            collapsed = projector @ rho @ projector / safe
            branch = (1 - kappa) * rho + kappa * collapsed
            branch = anp.where(valid, branch, anp.zeros_like(rho))
            branches.append(branch)
            defined.append(valid)
            if ordinary(born) and not bool(valid) and audit is not None:
                audit.append({
                    'kind': 'undefined_zero_probability_branch',
                    'born_probability': scalar_value(born),
                })
        return SelectiveMeasurementResult(
            probabilities, anp.stack(branches), anp.stack(defined), anp.stack(born_values))


def standard_profile():
    return SemanticProfile('standard_quantum')


def escort_profile(alpha):
    name = f'escort_born_{scalar_value(alpha):g}' if ordinary(alpha) else 'escort_born_trainable'
    return SemanticProfile(name, measurement_exponent=alpha)


def partial_collapse_profile(kappa):
    name = f'partial_collapse_{scalar_value(kappa):g}' if ordinary(kappa) else 'partial_collapse_trainable'
    return SemanticProfile(name, collapse_strength=kappa)


def spectral_power_profile(beta):
    name = f'spectral_power_{scalar_value(beta):g}' if ordinary(beta) else 'spectral_power_trainable'
    return SemanticProfile(name, evolution_spectral_power=beta)


def counterfactual_profile(name='counterfactual', measurement_exponent=2.,
                           collapse_strength=1., evolution_spectral_power=1.,
                           numerical_policy=None):
    return SemanticProfile(
        name, measurement_exponent, collapse_strength, evolution_spectral_power,
        numerical_policy=numerical_policy or NumericalPolicy())