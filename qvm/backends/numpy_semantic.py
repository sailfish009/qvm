"""Dense differentiable executor parameterized by explicit semantics and numerical policy."""
import numpy as np
from pennylane import numpy as anp
from ..numerics import UndefinedBranch,ordinary
I2=anp.eye(2,dtype=complex);X=anp.array([[0,1],[1,0]],complex);Y=anp.array([[0,-1j],[1j,0]],complex);Z=anp.array([[1,0],[0,-1]],complex);SWAP=anp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],complex)
def density(r):return (I2+r[0]*X+r[1]*Y+r[2]*Z)/2
class NumpySemanticBackend:
 name='numpy_semantic'
 def execute(self,program,inputs,semantics,return_trace=False):
  env={};trace=[];evolution_event=0;policy=semantics.numerical_policy
  for index,ins in enumerate(program.instructions):
   op=ins.op;a=[env[x] for x in ins.inputs];used=[];interventions=[];event=None
   if op=='input':v=inputs[ins.output]
   elif op=='density_from_bloch':v=density(a[0]);semantics.validate_state(v);used=['state_space']
   elif op=='ry_product_state':
    v=anp.array([1.+0j])
    for t in a[0]:v=anp.kron(v,anp.array([anp.cos(t/2),anp.sin(t/2)],complex))
    used=['state_space','composition_rule']
   elif op=='statevector_density':v=anp.outer(a[0],anp.conj(a[0]));used=['state_space']
   elif op=='pure_fidelity':v=anp.real(anp.vdot(a[0],a[1])*anp.conj(anp.vdot(a[0],a[1])))
   elif op=='tensor':v=semantics.compose(a[0],a[1],interventions);used=['composition_rule']
   elif op=='unitary':v=semantics.unitary(a[0],a[1],interventions);used=['evolution_spectral_power','evolution_application_scope','numerical_policy'];event=evolution_event;evolution_event+=1
   elif op=='partial_swap':
    t=a[1];u=anp.cos(t)*anp.eye(4)-1j*anp.sin(t)*SWAP;v=semantics.unitary(a[0],u,interventions);used=['evolution_spectral_power','evolution_application_scope','numerical_policy'];event=evolution_event;evolution_event+=1
   elif op=='partial_trace_second':z=anp.reshape(a[0],(2,2,2,2));v=z[:,0,:,0]+z[:,1,:,1]
   elif op=='dephase':v=semantics.channel(a[0],anp.stack((anp.array([[1,0],[0,0]],complex),anp.array([[0,0],[0,1]],complex))),interventions);used=['evolution_spectral_power','evolution_application_scope','numerical_policy'];event=evolution_event;evolution_event+=1
   elif op=='kraus_channel':v=semantics.channel(a[0],a[1],interventions);used=['evolution_spectral_power','evolution_application_scope','numerical_policy'];event=evolution_event;evolution_event+=1
   elif op=='povm_probabilities':v=semantics.probabilities(a[0],a[1],interventions);used=['measurement_exponent','numerical_policy']
   elif op=='projective_measurement':v=semantics.selective_projective_measurement(a[0],a[1],interventions);used=['measurement_exponent','collapse_strength','numerical_policy']
   elif op=='measurement_probabilities':v=a[0].probabilities
   elif op=='measurement_branches':v=a[0].branches
   elif op=='branch_state':
    outcome=int(ins.attrs['outcome']);defined=a[0].defined[outcome]
    if ordinary(defined) and not bool(np.asarray(defined)):raise UndefinedBranch(f'outcome {outcome} has zero Born probability; its conditional state is undefined')
    v=a[0].branches[outcome]
   elif op=='uhlmann_fidelity':
    sr=policy.sqrt(a[0],interventions,'uhlmann_left_sqrt');v=anp.real(anp.trace(policy.sqrt(sr@a[1]@sr,interventions,'uhlmann_middle_sqrt')))**2;used=['numerical_policy']
   elif op=='stabilize_fidelity':
    eps=float(ins.attrs['eps']);v=(a[0]+eps)/(1+eps);interventions.append({'kind':'declared_fidelity_stabilization','epsilon':eps});used=['numerical_policy']
   elif op=='bures_squared':v=2*(1-anp.sqrt(anp.maximum(a[0],0)));used=['numerical_policy']
   elif op=='sandwiched_renyi':
    alpha=float(ins.attrs['alpha'])
    if abs(alpha-1)<1e-12:v=anp.real(anp.trace(a[0]@(policy.log(a[0],interventions,'relative_entropy_log_rho')-policy.log(a[1],interventions,'relative_entropy_log_sigma'))))
    else:
     p=(1-alpha)/(2*alpha);left=policy.power(a[1],p,interventions,'renyi_sigma_power');m=left@a[0]@left;v=anp.log(anp.real(anp.trace(policy.power(m,alpha,interventions,'renyi_sandwiched_power'))))/(alpha-1)
    used=['numerical_policy']
   elif op=='pauli_readout':v=anp.stack(tuple(anp.real(anp.trace(a[0]@p)) for p in (X,Y,Z)))
   else:raise NotImplementedError(f'{self.name}: {op}')
   env[ins.output]=v;entry={'index':index,'output':ins.output,'op':op,'role':ins.role,'semantics':semantics.name,'assumptions_used':used,'numerical_interventions':interventions}
   if event is not None:entry['semantic_event_index']=event
   trace.append(entry)
  result=env[program.output];return (result,trace) if return_trace else result
