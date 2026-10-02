"""Deterministic semantic property probes that preserve counterexamples as data."""
from dataclasses import dataclass,asdict
import numpy as np
from .ir import Program
from .programs import povm_measurement,projective_measurement,unitary_evolution
from .semantics import standard_profile,escort_profile,partial_collapse_profile,spectral_power_profile

@dataclass(frozen=True)
class PropertyResult:
 property:str;classification:str;profile:str;status:str;delta:float;tolerance:float;witness:dict
 def record(self):return asdict(self)

def _identity_program(repetitions):
 p=Program(f'identity_evolution_{repetitions}');p.input('r','bloch',[3]);p.input('unitary','unitary',[2,2]);p.emit('rho','density_from_bloch','r',role='state');last='rho'
 for i in range(repetitions):last=p.emit(f'evolved_{i}','unitary',last,'unitary',role='evolution',event=i)
 return p.returns(last)
def _bloch(rho):
 X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1,-1]);return np.array([np.trace(rho@x).real for x in (X,Y,Z)])
def _record(name,classification,profile,delta,tolerance,witness):return PropertyResult(name,classification,profile.name,'preserved' if delta<=tolerance else 'counterexample',float(delta),float(tolerance),witness).record()

def inspect_profile(vm,profile,tolerance=1e-12):
 out=[];I=np.eye(2,dtype=complex);P=np.array([np.diag([1.,0.]),np.diag([0.,1.])]);zero=np.zeros(3)
 coarse=np.asarray(vm.run(povm_measurement(),{'r':zero,'effects':P},profile));refined_effects=np.array([P[0]/2,P[0]/2,P[1]]);fine=np.asarray(vm.run(povm_measurement(),{'r':zero,'effects':refined_effects},profile));simplex_delta=max(abs(float(coarse.sum()-1)),abs(float(fine.sum()-1)),max(0.,-float(coarse.min())));out.append(_record('probability_simplex','enforced_invariant',profile,simplex_delta,tolerance,{'coarse_probabilities':coarse.tolist(),'refined_probabilities':fine.tolist()}));delta=abs(float(coarse[0]-(fine[0]+fine[1])));out.append(_record('measurement_outcome_refinement','exploratory_property',profile,delta,tolerance,{'coarse_probabilities':coarse.tolist(),'refined_probabilities':fine.tolist(),'coarse_first':float(coarse[0]),'recombined_first':float(fine[0]+fine[1])}))
 r=np.array([0.,0.,.6]);p1=_identity_program(1);p2=_identity_program(2);a=np.asarray(vm.run(p1,{'r':r,'unitary':I},profile));b=np.asarray(vm.run(p2,{'r':r,'unitary':I},profile));state_delta=max(abs(float(np.trace(a).real-1)),float(np.max(abs(a-a.conj().T))),max(0.,-float(np.linalg.eigvalsh(a).min())));out.append(_record('density_state_validity','enforced_invariant',profile,state_delta,tolerance,{'one_step_diagonal':np.diag(a).real.tolist()}));delta=float(np.max(abs(a-b)));out.append(_record('identity_evolution_decomposition','exploratory_property',profile,delta,tolerance,{'input_bloch':r.tolist(),'one_step_diagonal':np.diag(a).real.tolist(),'two_step_diagonal':np.diag(b).real.tolist(),'one_step_program_sha256':p1.sha256,'two_step_program_sha256':p2.sha256,'application_scope':profile.evolution_application_scope}))
 r1=np.array([0.,0.,.8]);r2=np.array([0.,0.,-.2]);lam=.5;p=unitary_evolution();f1=np.asarray(vm.run(p,{'r':r1,'unitary':I},profile));f2=np.asarray(vm.run(p,{'r':r2,'unitary':I},profile));fm=np.asarray(vm.run(p,{'r':lam*r1+(1-lam)*r2,'unitary':I},profile));mix=lam*f1+(1-lam)*f2;delta=float(np.max(abs(fm-mix)));out.append(_record('convex_mixture_affinity','exploratory_property',profile,delta,tolerance,{'lambda':lam,'r1':r1.tolist(),'r2':r2.tolist(),'direct_diagonal':np.diag(fm).real.tolist(),'mixture_diagonal':np.diag(mix).real.tolist()}))
 measurement=vm.run(projective_measurement('full'),{'r':np.array([.6,0,.2]),'projectors':P},profile);branch=np.asarray(measurement.branches[0]);repeat=np.asarray(vm.run(povm_measurement(),{'r':_bloch(branch),'effects':P},profile));delta=abs(1-float(repeat[0]));out.append(_record('selective_measurement_repeatability','exploratory_property',profile,delta,tolerance,{'first_outcome':0,'branch_defined':bool(measurement.defined[0]),'repeat_probabilities':repeat.tolist()}))
 A=np.diag([.7,.3]);B=np.diag([.6,.4]);C=np.diag([.8,.2]);left=np.asarray(profile.compose(profile.compose(A,B),C));right=np.asarray(profile.compose(A,profile.compose(B,C)));delta=float(np.max(abs(left-right)));out.append(_record('tensor_associativity','held_fixed_invariant',profile,delta,tolerance,{'dimensions':[2,2,2]}))
 return out

def default_property_report(vm,tolerance=1e-12):
 profiles=(standard_profile(),escort_profile(4),partial_collapse_profile(.5),spectral_power_profile(2));results=[r for profile in profiles for r in inspect_profile(vm,profile,tolerance)];return {'schema':'qvm-property-report-v1','tolerance':tolerance,'profiles':[p.manifest() for p in profiles],'results':results,'counterexamples':[r for r in results if r['status']=='counterexample'],'interpretation':'Counterexamples characterize the declared profile; they are not silently discarded and do not by themselves establish or refute a law of nature.'}
