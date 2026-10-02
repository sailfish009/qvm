"""Explicit numerical-domain policy; numerical repair is not a physical semantic law."""
from dataclasses import dataclass,asdict
import numpy as np
from pennylane import numpy as anp

class NumericalDomainError(ValueError):pass
class UndefinedBranch(ValueError):pass

def ordinary(x):return not hasattr(x,'_value')
def scalar_value(x):return float(np.asarray(getattr(x,'_value',x)))

@dataclass(frozen=True)
class NumericalPolicy:
    name:str='support_preserving_float64'
    psd_roundoff_tolerance:float=1e-12
    probability_roundoff_tolerance:float=1e-14
    state_validation_tolerance:float=1e-9
    operator_validation_tolerance:float=1e-9
    bloch_validation_tolerance:float=1e-10
    singular_log_policy:str='raise'
    singular_negative_power_policy:str='raise'
    positive_power_zero_policy:str='preserve'
    probability_normalization:str='max_rescaled_power'
    def __post_init__(self):
        if not np.isfinite(self.psd_roundoff_tolerance) or self.psd_roundoff_tolerance<0:raise ValueError('invalid PSD tolerance')
        if not np.isfinite(self.probability_roundoff_tolerance) or self.probability_roundoff_tolerance<0:raise ValueError('invalid probability tolerance')
        for name in ('state_validation_tolerance','operator_validation_tolerance','bloch_validation_tolerance'):
            value=getattr(self,name)
            if not np.isfinite(value) or value<0:raise ValueError(f'invalid {name}')
        if self.singular_log_policy!='raise' or self.singular_negative_power_policy!='raise':raise NotImplementedError('v0.0003 implements strict singular matrix functions only')
        if self.positive_power_zero_policy!='preserve':raise NotImplementedError('v0.0003 preserves zero support for positive powers')
        if self.probability_normalization!='max_rescaled_power':raise NotImplementedError('unsupported probability normalization')
    def manifest(self):return asdict(self)
    def _spectrum(self,a,audit,operation):
        e,u=anp.linalg.eigh((a+anp.conj(a.T))/2);e=anp.real(e)
        if ordinary(e):
            values=np.asarray(e,dtype=float);minimum=float(values.min())
            if not np.isfinite(values).all():raise NumericalDomainError(f'{operation}: non-finite eigenvalue')
            if minimum < -self.psd_roundoff_tolerance:raise NumericalDomainError(f'{operation}: matrix is not positive semidefinite')
            count=int(np.sum(values<0))
            if count and audit is not None:audit.append({'kind':'negative_eigenvalue_roundoff_projection','operation':operation,'count':count,'minimum':minimum,'replacement':0.0})
        return anp.maximum(e,0),u
    def power(self,a,p,audit=None,operation='matrix_power'):
        if ordinary(p):
            exponent=scalar_value(p)
            if not np.isfinite(exponent):raise NumericalDomainError(f'{operation}: exponent must be finite')
        else:exponent=None
        e,u=self._spectrum(a,audit,operation)
        if exponent is not None and exponent<0:
            values=np.asarray(e,dtype=float)
            if np.any(values==0):raise NumericalDomainError(f'{operation}: negative power undefined on singular support')
        values=e**p
        return (u*values)@anp.conj(u.T)
    def sqrt(self,a,audit=None,operation='matrix_sqrt'):return self.power(a,.5,audit,operation)
    def log(self,a,audit=None,operation='matrix_log'):
        e,u=self._spectrum(a,audit,operation)
        if ordinary(e) and np.any(np.asarray(e,dtype=float)==0):raise NumericalDomainError(f'{operation}: logarithm undefined on singular support')
        return (u*anp.log(e))@anp.conj(u.T)
