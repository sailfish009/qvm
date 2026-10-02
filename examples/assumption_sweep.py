"""Counterfactual semantics audit. This does not infer laws of nature."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qvm import *
I=np.eye(2, dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1,-1]).astype(complex);PAULI=(X,Y,Z)
def effects(axis,sharpness):
 a=sum(axis[i]*PAULI[i] for i in range(3));return np.array([(I+sharpness*a)/2,(I-sharpness*a)/2])
def main():
 vm=QVM();rng=np.random.default_rng(20002);measure=povm_measurement();alphas=(1.,1.5,2.,3.,4.);tv={str(a):[] for a in alphas};normalization={str(a):0. for a in alphas}
 for _ in range(1000):
  r=rng.normal(size=3);r=r/np.linalg.norm(r)*rng.uniform(0,1);n=rng.normal(size=3);n/=np.linalg.norm(n);e=effects(n,rng.uniform(.1,1));x={'r':r,'effects':e};standard=np.asarray(vm.run(measure,x,standard_profile()))
  for a in alphas:
   p=np.asarray(vm.run(measure,x,escort_profile(a)));tv[str(a)].append(.5*np.abs(p-standard).sum());normalization[str(a)]=max(normalization[str(a)],abs(p.sum()-1))
 projectors=np.array([[[1,0],[0,0]],[[0,0],[0,1]]],complex);collapse={}
 for k in (0.,.25,.5,.75,1.):
  probs,branches=vm.run(projective_measurement('full'),{'r':np.array([.6,.2,.1]),'projectors':projectors},partial_collapse_profile(k));collapse[str(k)]={'probabilities':np.asarray(probs).tolist(),'branch0_bloch':[float(np.trace(branches[0]@o).real) for o in PAULI]}
 u=np.array([[1,1],[-1,1]],complex)/np.sqrt(2);spectral={}
 for b in (.5,1.,1.5,2.,4.):
  rho=np.asarray(vm.run(unitary_evolution(),{'r':np.array([.35,-.15,.4]),'unitary':u},spectral_power_profile(b)));spectral[str(b)]={'purity':float(np.trace(rho@rho).real),'eigenvalues':np.linalg.eigvalsh(rho).tolist()}
 profiles=[standard_profile(),escort_profile(1),partial_collapse_profile(.5),spectral_power_profile(2),counterfactual_profile('combined',1.5,.5,1.5)]
 report={'purpose':'counterfactual semantics coverage; not physical-law inference','cases':1000,'same_measurement_program_sha256':measure.sha256,'measurement_total_variation_from_standard':{a:{'mean':float(np.mean(v)),'max':float(np.max(v))} for a,v in tv.items()},'maximum_probability_normalization_error':normalization,'collapse_sweep':collapse,'spectral_power_sweep':spectral,'semantic_manifests':[p.manifest() for p in profiles],'coverage':{'implemented':['measurement exponent','selective-collapse strength','post-evolution spectral nonlinearity'],'held_standard':['complex density state space','tensor-product composition','convex input mixtures'],'not_covered':['alternative scalar fields','alternative composition laws','indefinite/signed states','nonlocal hidden variables','unknown unknowns']}}
 out=Path(__file__).resolve().parents[1]/'artifacts/assumption_sweep.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
