"""Replaceable semantic assumptions. Standard quantum theory is one profile, not a VM constant."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
from pennylane import numpy as anp

def _ordinary(x):return not hasattr(x,'_value')
def _hfunc(a,fn):
 e,u=anp.linalg.eigh((a+anp.conj(a.T))/2);e=anp.maximum(anp.real(e),1e-14);return (u*fn(e))@anp.conj(u.T)
def _spectral_power(rho,p):
 out=_hfunc(rho,lambda x:x**p);return out/anp.real(anp.trace(out))
@dataclass(frozen=True)
class SemanticProfile:
 name:str
 measurement_exponent:object=2.0
 collapse_strength:object=1.0
 evolution_spectral_power:object=1.0
 state_space:str='complex_density'
 composition_rule:str='tensor_product'
 mixture_rule:str='convex_linear'
 def _check(self):
  if self.state_space!='complex_density':raise NotImplementedError('alternative state spaces are not implemented in v0.0002')
  if self.composition_rule!='tensor_product':raise NotImplementedError('alternative composition rules are not implemented in v0.0002')
  if self.mixture_rule!='convex_linear':raise NotImplementedError('alternative input-mixture rules are not implemented in v0.0002')
  for key,value in (('measurement_exponent',self.measurement_exponent),('collapse_strength',self.collapse_strength),('evolution_spectral_power',self.evolution_spectral_power)):
   if _ordinary(value):
    x=float(value)
    if (key=='collapse_strength' and not 0<=x<=1) or (key!='collapse_strength' and x<=0):raise ValueError(f'invalid {key}')
 def manifest(self):
  self._check();alpha=float(self.measurement_exponent);k=float(self.collapse_strength);beta=float(self.evolution_spectral_power);changed=[]
  if alpha!=2:changed.append('born_exponent')
  if k!=1:changed.append('luders_collapse')
  if beta!=1:changed.append('linear_evolution')
  record={'schema':'qvm-semantics-v1','name':self.name,'state_space':self.state_space,'composition':self.composition_rule,'evolution':'linear_cptp' if beta==1 else 'spectral_power_after_map','measurement':'born' if alpha==2 else 'escort_born','measurement_exponent':alpha,'collapse':'luders' if k==1 else ('none' if k==0 else 'partial_luders_mix'),'collapse_strength':k,'mixture':self.mixture_rule,'evolution_spectral_power':beta,'changed_assumptions':changed,'enforced_invariants':['hermitian','positive_semidefinite','trace_one'],'known_nonstandard_properties':([] if not changed else self._violations(alpha,k,beta)),'numerical_policy':'eigenvalues below 1e-14 floored only for matrix powers; probability roundoff clamped at zero'}
  record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest();return record
 @staticmethod
 def _violations(alpha,k,beta):
  v=[]
  if alpha!=2:v+=['standard_born_rule','standard_noncontextual_measurement_assignment_not_assumed']
  if k!=1:v+=['standard_selective_measurement_update','repeatability_not_guaranteed']
  if beta!=1:v+=['linearity_of_state_evolution','convex_mixture_preservation_not_guaranteed']
  return v
 def validate_state(self,rho,tol=1e-9):
  if hasattr(rho,'_value'):return
  a=np.asarray(rho); 
  if a.ndim!=2 or a.shape[0]!=a.shape[1]:raise ValueError('state must be square')
  if not np.allclose(a,a.conj().T,atol=tol):raise ValueError('state must be Hermitian')
  if not np.isclose(np.trace(a),1,atol=tol):raise ValueError('state must have trace one')
  if np.linalg.eigvalsh(a).min()<-tol:raise ValueError('state must be positive semidefinite')
 def compose(self,a,b):return anp.kron(a,b)
 def after_evolution(self,rho):
  beta=self.evolution_spectral_power
  return rho if _ordinary(beta) and float(beta)==1 else _spectral_power(rho,beta)
 def unitary(self,rho,u):return self.after_evolution(u@rho@anp.conj(u.T))
 def channel(self,rho,kraus):return self.after_evolution(sum(k@rho@anp.conj(k.T) for k in kraus))
 def probabilities(self,rho,effects):
  raw=anp.stack(tuple(anp.real(anp.trace(rho@e)) for e in effects));raw=anp.maximum(raw,0);power=self.measurement_exponent/2;w=raw**power;return w/anp.sum(w)
 def selective_projective_measurement(self,rho,projectors):
  probabilities=self.probabilities(rho,projectors);branches=[];k=self.collapse_strength
  for i,p in enumerate(projectors):
   born=anp.real(anp.trace(rho@p));collapsed=p@rho@p/anp.maximum(born,1e-14);branches.append((1-k)*rho+k*collapsed)
  return probabilities,anp.stack(branches)

def standard_profile():return SemanticProfile('standard_quantum')
def escort_profile(alpha):return SemanticProfile(f'escort_born_{float(alpha):g}' if _ordinary(alpha) else 'escort_born_trainable',measurement_exponent=alpha)
def partial_collapse_profile(kappa):return SemanticProfile(f'partial_collapse_{float(kappa):g}' if _ordinary(kappa) else 'partial_collapse_trainable',collapse_strength=kappa)
def spectral_power_profile(beta):return SemanticProfile(f'spectral_power_{float(beta):g}' if _ordinary(beta) else 'spectral_power_trainable',evolution_spectral_power=beta)
def counterfactual_profile(name='counterfactual',measurement_exponent=2.,collapse_strength=1.,evolution_spectral_power=1.):return SemanticProfile(name,measurement_exponent,collapse_strength,evolution_spectral_power)
