"""Virtual machine: immutable Program, SemanticProfile and Backend are independent inputs."""
from dataclasses import dataclass
import hashlib
import numpy as np
from .backends.numpy_semantic import NumpySemanticBackend
from .backends.pennylane_standard import PennyLaneStandardBackend
from .semantics import standard_profile,SemanticProfile
@dataclass
class ExecutionResult:
 value:object;program_sha256:str;program_structural_sha256:str;semantics:dict;backend:str;trace:list;input_sha256:dict;used_assumptions:list;unused_changed_assumptions:list;numerical_interventions:list
 def audit_record(self):return {'program_sha256':self.program_sha256,'program_structural_sha256':self.program_structural_sha256,'semantics':self.semantics,'backend':self.backend,'trace':self.trace,'input_sha256':self.input_sha256,'used_assumptions':self.used_assumptions,'unused_changed_assumptions':self.unused_changed_assumptions,'numerical_interventions':self.numerical_interventions}
class QVM:
 def __init__(self):self.backends={'numpy_semantic':NumpySemanticBackend(),'pennylane_standard':PennyLaneStandardBackend()}
 def register_backend(self,name,backend):self.backends[name]=backend
 def _profile(self,x):
  if x is None or x=='standard':return standard_profile()
  if isinstance(x,SemanticProfile):return x
  raise TypeError('semantics must be a SemanticProfile or standard')
 @staticmethod
 def _hash(value):
  a=np.asarray(value);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()
 def _validate(self,program,inputs,profile):
  if not getattr(program,'sealed',False):raise ValueError('program must be sealed before execution')
  for ins in program.instructions:
   if ins.op!='input':continue
   if ins.output not in inputs:raise KeyError(f'missing input {ins.output}')
   v=inputs[ins.output]
   if hasattr(v,'_value'):continue
   a=np.asarray(v)
   if a.dtype.kind in 'fc' and not np.isfinite(a).all():raise ValueError(f'{ins.output} must be finite')
   kind=ins.attrs['qtype'];shape=ins.attrs.get('shape')
   if shape is not None and tuple(np.shape(v))!=tuple(shape):raise ValueError(f'{ins.output} expected shape {tuple(shape)}')
   if kind=='bloch' and np.linalg.norm(np.asarray(v,dtype=float))>1+profile.numerical_policy.bloch_validation_tolerance:raise ValueError('Bloch input outside unit ball for this profile')
   if kind=='unitary':
    u=np.asarray(v)
    if not np.allclose(u.conj().T@u,np.eye(u.shape[0]),atol=profile.numerical_policy.operator_validation_tolerance):raise ValueError('nonunitary input')
   if kind in ('effects','projectors'):
    es=np.asarray(v);n=es.shape[-1]
    if es.ndim!=3 or es.shape[-2:]!=(n,n):raise ValueError('invalid effects')
    tol=profile.numerical_policy.operator_validation_tolerance
    if not np.allclose(es,es.conj().transpose(0,2,1),atol=tol) or min(np.linalg.eigvalsh(es).reshape(-1))<-tol or not np.allclose(es.sum(0),np.eye(n),atol=tol):raise ValueError('effects must be a positive complete POVM')
    if kind=='projectors' and any(not np.allclose(e@e,e,atol=tol) for e in es):raise ValueError('projectors must be idempotent')
   if kind=='kraus':
    ks=np.asarray(v);total=sum(k.conj().T@k for k in ks)
    if not np.allclose(total,np.eye(total.shape[0]),atol=profile.numerical_policy.operator_validation_tolerance):raise ValueError('Kraus operators are not trace preserving')
 def run(self,program,inputs,semantics=None,backend='numpy_semantic',audit=False):
  profile=self._profile(semantics);profile._check()
  if backend not in self.backends:raise KeyError(f'unknown backend {backend}')
  self._validate(program,inputs,profile);value,trace=self.backends[backend].execute(program,inputs,profile,True)
  output_instruction=next(x for x in program.instructions if x.output==program.output)
  if output_instruction.role in ('state','evolution'):profile.validate_state(value)
  if not audit:return value
  hashes={k:self._hash(v) for k,v in inputs.items() if not hasattr(v,'_value')};manifest=profile.manifest();used_set={a for item in trace for a in item.get('assumptions_used',[])};used_set.add('numerical_policy');used=sorted(used_set)
  mapping={'born_exponent':'measurement_exponent','luders_collapse':'collapse_strength','linear_evolution':'evolution_spectral_power'};unused=[x for x in manifest['changed_assumptions'] if mapping[x] not in used]
  interventions=[{'instruction_index':item.get('index'),'op':item.get('op'),**event} for item in trace for event in item.get('numerical_interventions',[])]
  return ExecutionResult(value,program.sha256,program.structural_sha256,manifest,backend,trace,hashes,used,unused,interventions)
