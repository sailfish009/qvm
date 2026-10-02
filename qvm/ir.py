"""Assumption-neutral instruction tape for qvm_v0.0002."""
from dataclasses import dataclass,field
import hashlib,json
ROLES=('data','state','composition','evolution','measurement','host_math','output')
@dataclass(frozen=True)
class Instruction:
 output:str;op:str;inputs:tuple[str,...]=();role:str='host_math';attrs:dict=field(default_factory=dict)
 def __post_init__(self):
  if self.role not in ROLES:raise ValueError(f'unknown semantic role {self.role}')
 def record(self):return {'output':self.output,'op':self.op,'inputs':list(self.inputs),'role':self.role,'attrs':self.attrs}
class Program:
 def __init__(self,name):self.name=name;self.instructions=[];self.output=None;self._sealed=False
 def emit(self,output,op,*inputs,role='host_math',**attrs):
  if self._sealed:raise RuntimeError('program is sealed')
  known={x.output for x in self.instructions}
  if output in known:raise ValueError(f'duplicate value {output}')
  if op!='input' and any(x not in known for x in inputs):raise ValueError('input used before definition')
  self.instructions.append(Instruction(output,op,tuple(inputs),role,attrs));return output
 def input(self,name,qtype,shape=None):return self.emit(name,'input',role='data',qtype=qtype,shape=shape)
 def returns(self,value):
  if value not in {x.output for x in self.instructions}:raise ValueError('unknown return value')
  self.output=value;self._sealed=True;return self
 def record(self):
  if self.output is None:raise ValueError('program has no return')
  return {'schema':'qvm_v0.0002','name':self.name,'instructions':[x.record() for x in self.instructions],'return':self.output}
 def to_json(self,indent=None):return json.dumps(self.record(),sort_keys=True,separators=(',',':') if indent is None else None,indent=indent)
 @property
 def sha256(self):return hashlib.sha256(self.to_json().encode()).hexdigest()
 @classmethod
 def from_json(cls,text):
  r=json.loads(text)
  if r.get('schema')!='qvm_v0.0002':raise ValueError('wrong schema')
  p=cls(r['name'])
  for x in r['instructions']:p.emit(x['output'],x['op'],*x['inputs'],role=x['role'],**x['attrs'])
  return p.returns(r['return'])
