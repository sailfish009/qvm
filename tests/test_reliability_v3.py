import json,sys,unittest
from pathlib import Path
import numpy as np
import pennylane as qml
from pennylane import numpy as anp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qvm import *
I=np.eye(2,dtype=complex);P=np.array([np.diag([1.,0.]),np.diag([0.,1.])])
class ReliabilityTests(unittest.TestCase):
 def setUp(self):self.vm=QVM()
 def test_positive_power_preserves_exact_zero_support(self):
  rho=np.diag([1.,0.])
  for beta in (.1,.5,1.,2.,100.):np.testing.assert_array_equal(np.asarray(spectral_power_profile(beta).after_evolution(rho)),rho)
  d=self.vm.run(mixed_bures(),{'r':np.array([0.,0.,1.]),'s':np.array([0.,0.,-1.])});self.assertEqual(float(d),2.)
 def test_tiny_positive_branch_normalizes_and_zero_branch_is_undefined(self):
  z=1-1e-14;full=self.vm.run(projective_measurement('full'),{'r':np.array([0.,0.,z]),'projectors':P});self.assertTrue(bool(full.defined[1]));self.assertAlmostEqual(float(np.trace(full.branches[1]).real),1.,places=15)
  zero=self.vm.run(projective_measurement('full'),{'r':np.array([0.,0.,1.]),'projectors':P});self.assertFalse(bool(zero.defined[1]));np.testing.assert_array_equal(zero.branches[1],np.zeros((2,2)))
  with self.assertRaises(UndefinedBranch):self.vm.run(projective_measurement('branch',1),{'r':np.array([0.,0.,1.]),'projectors':P})
 def test_extreme_exponent_is_stable_and_nonfinite_parameters_rejected(self):
  got=self.vm.run(povm_measurement(),{'r':np.zeros(3),'effects':P},escort_profile(10000));np.testing.assert_array_equal(got,np.array([.5,.5]))
  for bad in (np.nan,np.inf,-np.inf):
   for factory in (escort_profile,spectral_power_profile):
    with self.assertRaises(ValueError):self.vm.run(povm_measurement(),{'r':np.zeros(3),'effects':P},factory(bad))
  for bad in (np.nan,np.inf,-np.inf):
   with self.assertRaises(ValueError):partial_collapse_profile(bad)._check()
 def test_singular_log_and_negative_power_are_explicit_domain_errors(self):
  x={'r':np.array([0.,0.,1.]),'s':np.array([0.,0.,.2])}
  with self.assertRaises(NumericalDomainError):self.vm.run(sandwiched_renyi(1),x)
  with self.assertRaises(NumericalDomainError):self.vm.run(sandwiched_renyi(2),{'r':np.array([0.,0.,.2]),'s':np.array([0.,0.,1.])})
 def test_sealed_program_is_deeply_immutable(self):
  p=partial_swap();before=p.to_json();record=p.record();record['instructions'][0]['attrs']['qtype']='changed';self.assertEqual(p.to_json(),before)
  actions=(lambda:p.instructions.append(None),lambda:p.instructions[0].attrs.__setitem__('x',1),lambda:p.returns('r'),lambda:setattr(p,'name','x'),lambda:setattr(p,'output','r'))
  for action in actions:
   with self.assertRaises((AttributeError,RuntimeError,TypeError)):action()
  self.assertEqual(p.to_json(),before)
 def test_structural_lowering_rejects_name_spoof_and_accepts_renaming(self):
  p=Program('partial_swap');p.input('r','bloch');p.input('s','bloch');p.input('tau','scalar');p.emit('rho','density_from_bloch','r',role='state');p.returns('rho');x={'r':np.array([.1,.2,.3]),'s':np.array([-.2,.1,0.]),'tau':.7}
  with self.assertRaises(UnsupportedLowering):self.vm.run(p,x,backend='pennylane_standard')
  canonical=partial_swap();record=canonical.record();record['name']='arbitrary_label';renamed=Program.from_json(json.dumps(record));self.assertEqual(canonical.structural_sha256,renamed.structural_sha256);np.testing.assert_allclose(self.vm.run(canonical,x,backend='pennylane_standard'),self.vm.run(renamed,x,backend='pennylane_standard'),atol=1e-15)
  custom=SemanticProfile('custom_numerics',numerical_policy=NumericalPolicy(state_validation_tolerance=1e-8))
  with self.assertRaises(UnsupportedLowering):self.vm.run(canonical,x,custom,backend='pennylane_standard')
 def test_used_and_unused_assumptions_are_distinct(self):
  result=self.vm.run(mixed_bures(),{'r':np.array([.1,0,0]),'s':np.array([0,.2,0])},escort_profile(4),audit=True);self.assertIn('born_exponent',result.unused_changed_assumptions);self.assertNotIn('measurement_exponent',result.used_assumptions);self.assertIn('numerical_policy',result.used_assumptions)
  measured=self.vm.run(povm_measurement(),{'r':np.zeros(3),'effects':P},escort_profile(4),audit=True);self.assertIn('measurement_exponent',measured.used_assumptions);self.assertEqual(measured.unused_changed_assumptions,[])
 def test_numerical_interventions_are_audited(self):
  result=self.vm.run(unitary_evolution(),{'r':np.array([0.,0.,1+1e-14]),'unitary':I},spectral_power_profile(2),audit=True);self.assertTrue(any(x['kind']=='negative_eigenvalue_roundoff_projection' for x in result.numerical_interventions))
 def test_property_explorer_preserves_counterexamples(self):
  report=default_property_report(self.vm);found={(x['profile'],x['property']) for x in report['counterexamples']};self.assertIn(('escort_born_4','measurement_outcome_refinement'),found);self.assertIn(('partial_collapse_0.5','selective_measurement_repeatability'),found);self.assertIn(('spectral_power_2','identity_evolution_decomposition'),found);self.assertIn(('spectral_power_2','convex_mixture_affinity'),found);self.assertFalse(any(x['profile']=='standard_quantum' for x in report['counterexamples']))
 def test_old_json_is_imported_then_resealed_as_current(self):
  for schema in ('qvm_v0.0002','qvm_v0.0003','qvm_v0.0004'):
   text=unitary_evolution().to_json().replace('qvm_v0.0006',schema);p=Program.from_json(text);self.assertEqual(p.record()['schema'],'qvm_v0.0006');self.assertTrue(p.sealed)
 def test_trainable_spectral_gradient_on_full_rank_state(self):
  p=unitary_evolution();r=np.array([.2,-.1,.4])
  def loss(beta):return anp.real(self.vm.run(p,{'r':r,'unitary':I},spectral_power_profile(beta))[0,0])
  beta=anp.array(1.3,requires_grad=True);g=qml.grad(loss)(beta);e=1e-6;num=(loss(1.3+e)-loss(1.3-e))/(2*e);self.assertAlmostEqual(float(g),float(num),places=7)
if __name__=='__main__':unittest.main()
