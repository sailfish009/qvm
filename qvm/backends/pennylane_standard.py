"""Independent PennyLane lowering restricted to exact supported tapes and standard semantics."""
import numpy as np
import pennylane as qml
from .numpy_semantic import density,SWAP
class UnsupportedLowering(RuntimeError):pass
class PennyLaneStandardBackend:
 name='pennylane_standard'
 @staticmethod
 def _kind(program,inputs):
  from ..programs import product_ry_fidelity,partial_swap,kraus_channel
  candidates=[]
  if 'q' in inputs:
   try:candidates.append(('product_ry_fidelity',product_ry_fidelity(len(inputs['q']))))
   except Exception:pass
  candidates.extend((('partial_swap',partial_swap()),('kraus_channel',kraus_channel())))
  for kind,canonical in candidates:
   if program.structural_sha256==canonical.structural_sha256:return kind
  raise UnsupportedLowering('instruction tape, attributes, or return value has no exact PennyLane lowering')
 def execute(self,program,inputs,semantics,return_trace=False):
  manifest=semantics.manifest()
  if manifest['changed_assumptions']:raise UnsupportedLowering('PennyLane circuit backend implements only the declared standard semantics')
  from ..numerics import NumericalPolicy
  if manifest['numerical_policy']!=NumericalPolicy().manifest():raise UnsupportedLowering('PennyLane lowering requires the standard declared numerical policy')
  kind=self._kind(program,inputs);base={'lowering_match':'exact_structural_fingerprint','program_structure_sha256':program.structural_sha256,'numerical_interventions':[]}
  if kind=='product_ry_fidelity':
   q,k=np.asarray(inputs['q']),np.asarray(inputs['k']);dev=qml.device('default.qubit',wires=len(q))
   @qml.qnode(dev)
   def state(a):
    for i,t in enumerate(a):qml.RY(t,wires=i)
    return qml.state()
   result=float(abs(np.vdot(state(q),state(k)))**2);trace=[{**base,'device':'default.qubit','lowering':'two RY state preparations and overlap','assumptions_used':['state_space','composition_rule']}]
  elif kind=='partial_swap':
   r,s=np.asarray(inputs['r']),np.asarray(inputs['s']);t=float(inputs['tau']);joint=np.kron(np.asarray(density(r)),np.asarray(density(s)));u=np.cos(t)*np.eye(4)-1j*np.sin(t)*np.asarray(SWAP);dev=qml.device('default.mixed',wires=2)
   @qml.qnode(dev)
   def circuit():qml.QubitDensityMatrix(joint,wires=[0,1]);qml.QubitUnitary(u,wires=[0,1]);return qml.density_matrix(wires=[0])
   result=np.asarray(circuit());trace=[{**base,'device':'default.mixed','lowering':'QubitUnitary and reduced density','assumptions_used':['state_space','composition_rule','evolution_spectral_power','evolution_application_scope']}]
  elif kind=='kraus_channel':
   rho=np.asarray(density(inputs['r']));ks=[np.asarray(k) for k in inputs['kraus']];dev=qml.device('default.mixed',wires=1)
   @qml.qnode(dev)
   def circuit():qml.QubitDensityMatrix(rho,wires=0);qml.QubitChannel(ks,wires=0);return qml.density_matrix(wires=0)
   result=np.asarray(circuit());trace=[{**base,'device':'default.mixed','lowering':'simulator QubitChannel; not asserted QPU-native','assumptions_used':['state_space','evolution_spectral_power','evolution_application_scope']}]
  return (result,trace) if return_trace else result
