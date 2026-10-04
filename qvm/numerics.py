"""Explicit numerical-domain policy.

Numerical repair is a declared policy, not a physical semantic law. Every
projection or substitution must be recorded in the audit interventions.
"""
from dataclasses import dataclass, asdict
import numpy as np
from pennylane import numpy as anp


class NumericalDomainError(ValueError):
    pass


class UndefinedBranch(ValueError):
    pass


def ordinary(x):
    """True for plain NumPy/Python values, false for Autograd-traced values."""
    return not hasattr(x, '_value')


def scalar_value(x):
    return float(np.asarray(getattr(x, '_value', x)))


@dataclass(frozen=True)
class NumericalPolicy:
    name: str = 'support_preserving_float64'
    psd_roundoff_tolerance: float = 1e-12
    probability_roundoff_tolerance: float = 1e-14
    state_validation_tolerance: float = 1e-9
    operator_validation_tolerance: float = 1e-9
    bloch_validation_tolerance: float = 1e-10
    hermitian_roundoff_tolerance: float = 1e-10
    proper_time_cutoff: float = 40.0
    singular_log_policy: str = 'raise'
    singular_negative_power_policy: str = 'raise'
    positive_power_zero_policy: str = 'preserve'
    probability_normalization: str = 'max_rescaled_power'

    def __post_init__(self):
        for name in ('psd_roundoff_tolerance', 'probability_roundoff_tolerance',
                     'state_validation_tolerance', 'operator_validation_tolerance',
                     'bloch_validation_tolerance', 'hermitian_roundoff_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'invalid {name}')
        if not np.isfinite(self.proper_time_cutoff) or self.proper_time_cutoff <= 0:
            raise ValueError('proper_time_cutoff must be finite and positive')
        if self.singular_log_policy != 'raise' or self.singular_negative_power_policy != 'raise':
            raise NotImplementedError('v0.0005 implements strict singular matrix functions only')
        if self.positive_power_zero_policy != 'preserve':
            raise NotImplementedError('v0.0005 preserves zero support for positive powers')
        if self.probability_normalization != 'max_rescaled_power':
            raise NotImplementedError('unsupported probability normalization')

    def manifest(self):
        return asdict(self)

    def _spectrum(self, matrix, audit, operation):
        """Eigen-decompose an explicitly Hermitian matrix; never silently symmetrize."""
        raw = getattr(matrix, '_value', matrix)
        if ordinary(raw):
            values = np.asarray(raw)
            deviation = float(np.max(np.abs(values - values.conj().T))) if values.size else 0.0
            if deviation > self.hermitian_roundoff_tolerance:
                raise NumericalDomainError(
                    f'{operation}: matrix is not Hermitian (max deviation {deviation:g})')
        hermitian = (matrix + anp.conj(matrix.T)) / 2
        eigenvalues, eigenvectors = anp.linalg.eigh(hermitian)
        eigenvalues = anp.real(eigenvalues)
        if ordinary(eigenvalues):
            values = np.asarray(eigenvalues, dtype=float)
            if not np.isfinite(values).all():
                raise NumericalDomainError(f'{operation}: non-finite eigenvalue')
            minimum = float(values.min())
            if minimum < -self.psd_roundoff_tolerance:
                raise NumericalDomainError(f'{operation}: matrix is not positive semidefinite')
            count = int(np.sum(values < 0))
            if count and audit is not None:
                audit.append({
                    'kind': 'negative_eigenvalue_roundoff_projection',
                    'operation': operation,
                    'count': count,
                    'minimum': minimum,
                    'replacement': 0.0,
                })
        return anp.maximum(eigenvalues, 0), eigenvectors

    def power(self, matrix, exponent, audit=None, operation='matrix_power'):
        if ordinary(exponent):
            value = scalar_value(exponent)
            if not np.isfinite(value):
                raise NumericalDomainError(f'{operation}: exponent must be finite')
            defined_exponent = value
        else:
            defined_exponent = None
        eigenvalues, eigenvectors = self._spectrum(matrix, audit, operation)
        if defined_exponent is not None and defined_exponent < 0:
            if np.any(np.asarray(eigenvalues, dtype=float) == 0):
                raise NumericalDomainError(
                    f'{operation}: negative power undefined on singular support')
        powered = eigenvalues ** exponent
        return (eigenvectors * powered) @ anp.conj(eigenvectors.T)

    def sqrt(self, matrix, audit=None, operation='matrix_sqrt'):
        return self.power(matrix, 0.5, audit, operation)

    def log(self, matrix, audit=None, operation='matrix_log'):
        eigenvalues, eigenvectors = self._spectrum(matrix, audit, operation)
        if ordinary(eigenvalues) and np.any(np.asarray(eigenvalues, dtype=float) == 0):
            raise NumericalDomainError(f'{operation}: logarithm undefined on singular support')
        return (eigenvectors * anp.log(eigenvalues)) @ anp.conj(eigenvectors.T)