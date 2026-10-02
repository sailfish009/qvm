"""Deeply sealed assumption-neutral instruction tape for qvm_v0.0003."""
from dataclasses import dataclass,field
from types import MappingProxyType
from collections.abc import Mapping
import hashlib,json
ROLES=('data','state','composition','evolution','measurement','host_math','output')

def _freeze(x):
 if isinstance(x,Mapping):return MappingProxyType({str(k):_freeze(v) for k,v in x.items()})
 if isinstance(x,(list,tuple)):return tuple(_freeze(v) for v in x)
 return x

def _thaw(x):
 if isinstance(x,Mapping):return {k:_thaw(v) for k,v in x.items()}
 if isinstance(x,tuple):return [_thaw(v) for v in x]
 return x

@dataclass(frozen=True)
class Instruction:
 output:str;op:str;inputs:tuple[str,...]=();role:str='host_math';attrs:Mapping=field(default_factory=dict)
 def __post_init__(self):
  if self.role not in ROLES:raise ValueError(f'unknown semantic role {self.role}')
  object.__setattr__(self,'inputs',tuple(self.inputs));object.__setattr__(self,'attrs',_freeze(self.attrs))
 def record(self):return {'output':self.output,'op':self.op,'inputs':list(self.inputs),'role':self.role,'attrs':_thaw(self.attrs)}

class Program:
 def __init__(self,name):
  if not isinstance(name,str) or not name:raise ValueError('program name must be nonempty')
  self._name=name;self._instructions=[];self._output=None;self._sealed=False
 @property
 def name(self):return self._name
 @property
 def instructions(self):return self._instructions if not self._sealed else tuple(self._instructions)
 @property
 def output(self):return self._output
 @property
 def sealed(self):return self._sealed
 def emit(self,output,op,*inputs,role='host_math',**attrs):
  if self._sealed:raise RuntimeError('program is sealed')
  known={x.output for x in self._instructions}
  if output in known:raise ValueError(f'duplicate value {output}')
  if op!='input' and any(x not in known for x in inputs):raise ValueError('input used before definition')
  self._instructions.append(Instruction(output,op,tuple(inputs),role,attrs));return output
 def input(self,name,qtype,shape=None):return self.emit(name,'input',role='data',qtype=qtype,shape=shape)
 def returns(self,value):
  if self._sealed:raise RuntimeError('program is sealed')
  if value not in {x.output for x in self._instructions}:raise ValueError('unknown return value')
  self._output=value;self._instructions=tuple(self._instructions);self._sealed=True;return self
 def record(self):
  if not self._sealed or self._output is None:raise ValueError('program is not sealed')
  return {'schema':'qvm_v0.0003','name':self._name,'instructions':[x.record() for x in self._instructions],'return':self._output}
 def structural_record(self):
  r=self.record();return {'instructions':r['instructions'],'return':r['return']}
 def to_json(self,indent=None):return json.dumps(self.record(),sort_keys=True,separators=(',',':') if indent is None else None,indent=indent)
 @property
 def sha256(self):return hashlib.sha256(self.to_json().encode()).hexdigest()
 @property
 def structural_sha256(self):return hashlib.sha256(json.dumps(self.structural_record(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
 @classmethod
 def from_json(cls,text):
  r=json.loads(text)
  if r.get('schema') not in ('qvm_v0.0002','qvm_v0.0003'):raise ValueError('wrong schema')
  p=cls(r['name'])
  for x in r['instructions']:p.emit(x['output'],x['op'],*x['inputs'],role=x['role'],**x['attrs'])
  return p.returns(r['return'])
