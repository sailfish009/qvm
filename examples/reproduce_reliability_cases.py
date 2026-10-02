"""Reproduce the public-review reliability cases fixed in version 0.0.3."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qvm import *
ROOT=Path(__file__).resolve().parents[1]
def main():
 vm=QVM();P=np.array([np.diag([1.,0.]),np.diag([0.,1.])]);pure=np.diag([1.,0.]);spectral=np.asarray(spectral_power_profile(.1).after_evolution(pure));z=1-1e-14;measurement=vm.run(projective_measurement('full'),{'r':np.array([0.,0.,z]),'projectors':P});extreme=np.asarray(vm.run(povm_measurement(),{'r':np.zeros(3),'effects':P},escort_profile(10000)))
 rejected=[]
 for value in (float('nan'),float('inf'),float('-inf')):
  try:escort_profile(value)._check()
  except ValueError:rejected.append(str(value))
 spoof=Program('partial_swap');spoof.input('r','bloch');spoof.input('s','bloch');spoof.input('tau','scalar');spoof.emit('rho','density_from_bloch','r',role='state');spoof.returns('rho');lowering_rejected=False
 try:vm.run(spoof,{'r':np.zeros(3),'s':np.zeros(3),'tau':.2},backend='pennylane_standard')
 except UnsupportedLowering:lowering_rejected=True
 sealed=partial_swap();mutation_rejected=False
 try:sealed.instructions[0].attrs['changed']=True
 except (AttributeError,TypeError):mutation_rejected=True
 report={'schema':'qvm-reliability-regressions-v1','version':'0.0.3','positive_power_pure_state_diagonal':np.diag(spectral).real.tolist(),'tiny_positive_branch_probability':float(measurement.born_probabilities[1]),'tiny_positive_branch_trace':float(np.trace(measurement.branches[1]).real),'extreme_equal_probability_alpha10000':extreme.tolist(),'nonfinite_parameters_rejected':rejected,'altered_tape_pennylane_rejected':lowering_rejected,'sealed_attribute_mutation_rejected':mutation_rejected}
 assert report['positive_power_pure_state_diagonal']==[1.,0.] and abs(report['tiny_positive_branch_trace']-1)<1e-15 and report['extreme_equal_probability_alpha10000']==[.5,.5] and len(rejected)==3 and lowering_rejected and mutation_rejected
 out=ROOT/'artifacts/reliability_regressions.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
