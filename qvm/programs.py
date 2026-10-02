"""Assumption-neutral reference programs."""
from .ir import Program
def product_ry_fidelity(n=12):
 p=Program('product_ry_fidelity');p.input('q','angles',[n]);p.input('k','angles',[n]);p.emit('qs','ry_product_state','q',role='state');p.emit('ks','ry_product_state','k',role='state');p.emit('f','pure_fidelity','qs','ks',role='host_math');return p.returns('f')
def povm_measurement():
 p=Program('povm_measurement');p.input('r','bloch',[3]);p.input('effects','effects',None);p.emit('rho','density_from_bloch','r',role='state');p.emit('probabilities','povm_probabilities','rho','effects',role='measurement');return p.returns('probabilities')
def projective_measurement(return_part='full',outcome=0):
 p=Program('projective_measurement');p.input('r','bloch',[3]);p.input('projectors','projectors',None);p.emit('rho','density_from_bloch','r',role='state');p.emit('measurement','projective_measurement','rho','projectors',role='measurement')
 if return_part=='probabilities':p.emit('out','measurement_probabilities','measurement',role='output')
 elif return_part=='branches':p.emit('out','measurement_branches','measurement',role='output')
 elif return_part=='branch':p.emit('out','branch_state','measurement',role='output',outcome=int(outcome))
 elif return_part=='full':return p.returns('measurement')
 else:raise ValueError(return_part)
 return p.returns('out')
def unitary_evolution():
 p=Program('unitary_evolution');p.input('r','bloch',[3]);p.input('unitary','unitary',[2,2]);p.emit('rho','density_from_bloch','r',role='state');p.emit('out','unitary','rho','unitary',role='evolution');return p.returns('out')
def partial_swap():
 p=Program('partial_swap');p.input('r','bloch',[3]);p.input('s','bloch',[3]);p.input('tau','scalar',[]);p.emit('rho','density_from_bloch','r',role='state');p.emit('sigma','density_from_bloch','s',role='state');p.emit('joint','tensor','rho','sigma',role='composition');p.emit('evolved','partial_swap','joint','tau',role='evolution');p.emit('out','partial_trace_second','evolved',role='state');return p.returns('out')
def mixed_bures(stabilization_eps=0.):
 p=Program('mixed_bures');p.input('r','bloch',[3]);p.input('s','bloch',[3]);p.emit('rho','density_from_bloch','r',role='state');p.emit('sigma','density_from_bloch','s',role='state');p.emit('f0','uhlmann_fidelity','rho','sigma',role='host_math');f='f0'
 if stabilization_eps:p.emit('f','stabilize_fidelity','f0',role='host_math',eps=float(stabilization_eps));f='f'
 p.emit('d','bures_squared',f,role='host_math');return p.returns('d')
def sandwiched_renyi(alpha=.9):
 p=Program(f'sandwiched_renyi_{alpha:g}');p.input('r','bloch',[3]);p.input('s','bloch',[3]);p.emit('rho','density_from_bloch','r',role='state');p.emit('sigma','density_from_bloch','s',role='state');p.emit('d','sandwiched_renyi','rho','sigma',role='host_math',alpha=float(alpha));return p.returns('d')
def kraus_channel():
 p=Program('kraus_channel');p.input('r','bloch',[3]);p.input('kraus','kraus',None);p.emit('rho','density_from_bloch','r',role='state');p.emit('out','kraus_channel','rho','kraus',role='evolution');return p.returns('out')
