"""Reproduce frozen standard-profile outputs without external repository dependencies."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qvm import QVM,standard_profile,product_ry_fidelity,mixed_bures,sandwiched_renyi,partial_swap
ROOT=Path(__file__).resolve().parents[1]
def blocks(x):
 x=x.reshape(4,3);norm=np.sqrt(np.sum(x*x,axis=-1,keepdims=True)+1e-12);return x/norm*np.tanh(norm)
def main():
 manifest=json.loads((ROOT/'reference/MANIFEST.json').read_text());blob=ROOT/'reference/standard_vectors.npz';assert hashlib.sha256(blob.read_bytes()).hexdigest()==manifest['npz_sha256'];vm=QVM();profile=standard_profile();err={'product_fidelity':0.,'mixed_bures':0.,'renyi_09':0.,'partial_swap':0.};hp=manifest['historical_parameters']
 with np.load(blob) as d:
  for i in range(manifest['cases']):
   q,k=d['q'][i],d['k'][i];new=float(vm.run(product_ry_fidelity(),{'q':q,'k':k},profile));err['product_fidelity']=max(err['product_fidelity'],abs(new-d['product_fidelity'][i]))
   rb,sb=blocks(q),blocks(k);new=-sum(float(vm.run(mixed_bures(hp['bures_eps']),{'r':rb[j],'s':sb[j]},profile)) for j in range(4));err['mixed_bures']=max(err['mixed_bures'],abs(new-d['mixed_bures_score'][i]))
   rb=(1-hp['renyi_density_eps'])*rb;sb=(1-hp['renyi_density_eps'])*sb;new=sum(float(vm.run(sandwiched_renyi(hp['renyi_alpha']),{'r':rb[j],'s':sb[j]},profile)) for j in range(4));err['renyi_09']=max(err['renyi_09'],abs(new-d['renyi_09'][i]))
   new=np.asarray(vm.run(partial_swap(),{'r':d['partial_r'][i],'s':d['partial_s'][i],'tau':d['partial_tau'][i]},profile));err['partial_swap']=max(err['partial_swap'],float(np.max(abs(new-d['partial_density'][i]))))
 report={'profile':profile.manifest(),'cases':manifest['cases'],'reference_manifest_sha256':hashlib.sha256((ROOT/'reference/MANIFEST.json').read_bytes()).hexdigest(),'maximum_error':err,'conditional_claim':'reproduction under the declared standard profile only'};out=ROOT/'artifacts/standard_reproduction.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
