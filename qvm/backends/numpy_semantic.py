"""Dense differentiable executor parameterized by an explicit semantic profile."""
import numpy as np
from pennylane import numpy as anp
I2=anp.eye(2,dtype=complex);X=anp.array([[0,1],[1,0]],complex);Y=anp.array([[0,-1j],[1j,0]],complex);Z=anp.array([[1,0],[0,-1]],complex);SWAP=anp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)
def density(r):return (I2+r[0]*X+r[1]*Y+r[2]*Z)/2
def hfunc(a,fn):
 e,u=anp.linalg.eigh((a+anp.conj(a.T))/2);e=anp.maximum(anp.real(e),1e-14);return (u*fn(e))@anp.conj(u.T)
def hsqrt(a):return hfunc(a,anp.sqrt)
def hpow(a,p):return hfunc(a,lambda x:x**p)
def hlog(a):return hfunc(a,anp.log)
class NumpySemanticBackend:
 name='numpy_semantic'
 def execute(self,program,inputs,semantics,return_trace=False):
  env={};trace=[]
  for ins in program.instructions:
   op=ins.op;a=[env[x] for x in ins.inputs]
   if op=='input':v=inputs[ins.output]
   elif op=='density_from_bloch':v=density(a[0]);semantics.validate_state(v)
   elif op=='ry_product_state':
    v=anp.array([1.+0j])
    for t in a[0]:v=anp.kron(v,anp.array([anp.cos(t/2),anp.sin(t/2)],complex))
   elif op=='statevector_density':v=anp.outer(a[0],anp.conj(a[0]))
   elif op=='pure_fidelity':v=anp.real(anp.vdot(a[0],a[1])*anp.conj(anp.vdot(a[0],a[1])))
   elif op=='tensor':v=semantics.compose(a[0],a[1])
   elif op=='unitary':v=semantics.unitary(a[0],a[1])
   elif op=='partial_swap':
    t=a[1];u=anp.cos(t)*anp.eye(4)-1j*anp.sin(t)*SWAP;v=semantics.unitary(a[0],u)
   elif op=='partial_trace_second':
    z=anp.reshape(a[0],(2,2,2,2));v=z[:,0,:,0]+z[:,1,:,1]
   elif op=='dephase':v=semantics.channel(a[0],anp.stack((anp.array([[1,0],[0,0]],complex),anp.array([[0,0],[0,1]],complex))))
   elif op=='kraus_channel':v=semantics.channel(a[0],a[1])
   elif op=='povm_probabilities':v=semantics.probabilities(a[0],a[1])
   elif op=='projective_measurement':v=semantics.selective_projective_measurement(a[0],a[1])
   elif op=='measurement_probabilities':v=a[0][0]
   elif op=='measurement_branches':v=a[0][1]
   elif op=='branch_state':v=a[0][1][int(ins.attrs['outcome'])]
   elif op=='uhlmann_fidelity':
    sr=hsqrt(a[0]);v=anp.real(anp.trace(hsqrt(sr@a[1]@sr)))**2
   elif op=='stabilize_fidelity':
    eps=float(ins.attrs['eps']);v=(a[0]+eps)/(1+eps)
   elif op=='bures_squared':v=2*(1-anp.sqrt(anp.maximum(a[0],0)))
   elif op=='sandwiched_renyi':
    alpha=float(ins.attrs['alpha'])
    if abs(alpha-1)<1e-12:v=anp.real(anp.trace(a[0]@(hlog(a[0])-hlog(a[1]))))
    else:
     p=(1-alpha)/(2*alpha);m=hpow(a[1],p)@a[0]@hpow(a[1],p);v=anp.log(anp.real(anp.trace(hpow(m,alpha))))/(alpha-1)
   elif op=='pauli_readout':v=anp.stack(tuple(anp.real(anp.trace(a[0]@p)) for p in (X,Y,Z)))
   else:raise NotImplementedError(f'{self.name}: {op}')
   env[ins.output]=v;trace.append({'output':ins.output,'op':op,'role':ins.role,'semantics':semantics.name})
  result=env[program.output];return (result,trace) if return_trace else result
