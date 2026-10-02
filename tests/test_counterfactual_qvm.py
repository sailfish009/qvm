import sys,unittest,json
from pathlib import Path
import numpy as np
import pennylane as qml
from pennylane import numpy as anp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qvm import *
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1,-1]).astype(complex)
def trine():
 ns=[np.array([np.cos(2*np.pi*k/3),np.sin(2*np.pi*k/3),0.]) for k in range(3)];return np.array([(I+n[0]*X+n[1]*Y+n[2]*Z)/3 for n in ns])
P=np.array([[[1,0],[0,0]],[[0,0],[0,1]]],complex)
class Tests(unittest.TestCase):
 def setUp(self):self.vm=QVM();self.r=np.array([.43,-.21,.31]);self.effects=trine()
 def test_program_is_semantics_neutral_and_serializable(self):
  p=povm_measurement();q=Program.from_json(p.to_json());self.assertEqual(p.sha256,q.sha256)
  a=self.vm.run(p,{'r':self.r,'effects':self.effects},standard_profile(),audit=True);b=self.vm.run(p,{'r':self.r,'effects':self.effects},escort_profile(1),audit=True);self.assertEqual(a.program_sha256,b.program_sha256);self.assertNotEqual(a.semantics['sha256'],b.semantics['sha256']);self.assertNotEqual(a.value.tolist(),b.value.tolist())
 def test_standard_and_escort_measurement(self):
  p=povm_measurement();x={'r':self.r,'effects':self.effects};std=self.vm.run(p,x,standard_profile());a2=self.vm.run(p,x,escort_profile(2));a1=self.vm.run(p,x,escort_profile(1));a4=self.vm.run(p,x,escort_profile(4));np.testing.assert_allclose(std,a2,atol=1e-15)
  for z in (std,a1,a4):self.assertAlmostEqual(float(np.sum(z)),1.,places=14);self.assertTrue(np.all(np.asarray(z)>=0))
  self.assertGreater(np.linalg.norm(a1-std),1e-3);self.assertGreater(np.linalg.norm(a4-std),1e-3)
 def test_trainable_assumption_gradient(self):
  p=povm_measurement();target=np.array([.2,.3,.5])
  def loss(alpha):
   y=self.vm.run(p,{'r':self.r,'effects':self.effects},escort_profile(alpha));return anp.sum((y-target)**2)
  alpha=anp.array(1.7,requires_grad=True);g=qml.grad(loss)(alpha);e=1e-6;num=(loss(1.7+e)-loss(1.7-e))/(2*e);self.assertAlmostEqual(float(g),float(num),places=7)
 def test_partial_collapse_changes_update_not_probability(self):
  p=projective_measurement('full');x={'r':np.array([.6,0,.2]),'projectors':P};stdp,stdb=self.vm.run(p,x,standard_profile());nonep,noneb=self.vm.run(p,x,partial_collapse_profile(0));halfp,halfb=self.vm.run(p,x,partial_collapse_profile(.5));np.testing.assert_allclose(stdp,nonep);np.testing.assert_allclose(stdp,halfp)
  rho=(I+.6*X+.2*Z)/2;np.testing.assert_allclose(noneb[0],rho);np.testing.assert_allclose(stdb[0],P[0]);np.testing.assert_allclose(halfb[0],.5*(rho+P[0]));self.assertEqual(partial_collapse_profile(0).manifest()['changed_assumptions'],['luders_collapse'])
 def test_spectral_power_changes_linear_evolution(self):
  p=unitary_evolution();u=np.array([[1,1],[-1,1]],complex)/np.sqrt(2);x={'r':self.r,'unitary':u};a=self.vm.run(p,x,standard_profile());b=self.vm.run(p,x,spectral_power_profile(1));c=self.vm.run(p,x,spectral_power_profile(2));np.testing.assert_allclose(a,b,atol=1e-14);self.assertGreater(np.linalg.norm(c-a),1e-3);self.assertAlmostEqual(float(np.trace(c).real),1.,places=13);self.assertGreater(np.trace(c@c).real,np.trace(a@a).real)
 def test_standard_prior_equations_and_pennylane(self):
  rng=np.random.default_rng(2);q,k=rng.normal(size=(2,12));f=self.vm.run(product_ry_fidelity(),{'q':q,'k':k});self.assertAlmostEqual(float(f),float(np.prod(np.cos((q-k)/2)**2)),places=14)
  x={'r':self.r,'s':np.array([-.3,.2,.4]),'tau':.71};dense=self.vm.run(partial_swap(),x);pl=self.vm.run(partial_swap(),x,backend='pennylane_standard');np.testing.assert_allclose(dense,pl,atol=3e-15)
 def test_alternative_semantics_not_silently_lowered_to_pennylane(self):
  with self.assertRaises(UnsupportedLowering):self.vm.run(partial_swap(),{'r':self.r,'s':np.array([-.3,.2,.4]),'tau':.7},spectral_power_profile(2),backend='pennylane_standard')
 def test_profile_manifest_is_explicit(self):
  m=counterfactual_profile('combined',1.3,.4,1.7).manifest();self.assertEqual(set(m['changed_assumptions']),{'born_exponent','luders_collapse','linear_evolution'});self.assertIn('convex_mixture_preservation_not_guaranteed',m['known_nonstandard_properties']);self.assertEqual(len(m['sha256']),64)
 def test_input_validation_depends_on_declared_standard_state_space(self):
  with self.assertRaises(ValueError):self.vm.run(povm_measurement(),{'r':np.array([2.,0,0]),'effects':self.effects})
  bad=self.effects.copy();bad[0]*=2
  with self.assertRaises(ValueError):self.vm.run(povm_measurement(),{'r':self.r,'effects':bad})
 def test_program_role_classification(self):
  roles={x.role for x in projective_measurement().instructions};self.assertTrue({'data','state','measurement'}.issubset(roles))
  with self.assertRaises(NotImplementedError):self.vm.run(povm_measurement(),{'r':self.r,'effects':self.effects},SemanticProfile('unsupported',composition_rule='direct_sum'))
 def test_no_torch(self):self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
