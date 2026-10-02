"""Independent PennyLane lowering restricted to standard semantics."""
import numpy as np
import pennylane as qml
from .numpy_semantic import density,SWAP
class UnsupportedLowering(RuntimeError):pass
class PennyLaneStandardBackend:
 name='pennylane_standard'
 def execute(self,program,inputs,semantics,return_trace=False):
  manifest=semantics.manifest()
  if manifest['changed_assumptions']:raise UnsupportedLowering('PennyLane circuit backend implements only the declared standard semantics')
  ops=[x.op for x in program.instructions]
  if program.name=='product_ry_fidelity':
   q,k=np.asarray(inputs['q']),np.asarray(inputs['k']);dev=qml.device('default.qubit',wires=len(q))
   @qml.qnode(dev)
   def state(a):
    for i,t in enumerate(a):qml.RY(t,wires=i)
    return qml.state()
   result=float(abs(np.vdot(state(q),state(k)))**2);trace=[{'device':'default.qubit','lowering':'two RY state preparations and overlap'}]
  elif program.name.startswith('partial_swap'):
   r,s=np.asarray(inputs['r']),np.asarray(inputs['s']);t=float(inputs['tau']);joint=np.kron(np.asarray(density(r)),np.asarray(density(s)));u=np.cos(t)*np.eye(4)-1j*np.sin(t)*np.asarray(SWAP);dev=qml.device('default.mixed',wires=2)
   @qml.qnode(dev)
   def circuit():qml.QubitDensityMatrix(joint,wires=[0,1]);qml.QubitUnitary(u,wires=[0,1]);return qml.density_matrix(wires=[0])
   result=np.asarray(circuit());trace=[{'device':'default.mixed','lowering':'QubitUnitary and reduced density'}]
  elif program.name=='kraus_channel':
   rho=np.asarray(density(inputs['r']));ks=[np.asarray(k) for k in inputs['kraus']];dev=qml.device('default.mixed',wires=1)
   @qml.qnode(dev)
   def circuit():qml.QubitDensityMatrix(rho,wires=0);qml.QubitChannel(ks,wires=0);return qml.density_matrix(wires=0)
   result=np.asarray(circuit());trace=[{'device':'default.mixed','lowering':'simulator QubitChannel; not asserted QPU-native'}]
  else:raise UnsupportedLowering(f'{program.name} has no physical-circuit lowering')
  return (result,trace) if return_trace else result
