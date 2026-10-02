"""qvm_v0.0002: counterfactual semantic execution for quantum learning equations."""
from .ir import Program,Instruction
from .vm import QVM,ExecutionResult
from .semantics import SemanticProfile,standard_profile,escort_profile,partial_collapse_profile,spectral_power_profile,counterfactual_profile
from .programs import product_ry_fidelity,povm_measurement,projective_measurement,unitary_evolution,partial_swap,mixed_bures,sandwiched_renyi,kraus_channel
from .backends.pennylane_standard import UnsupportedLowering
__version__='0.0.2'
__all__=['Program','Instruction','QVM','ExecutionResult','SemanticProfile','standard_profile','escort_profile','partial_collapse_profile','spectral_power_profile','counterfactual_profile','product_ry_fidelity','povm_measurement','projective_measurement','unitary_evolution','partial_swap','mixed_bures','sandwiched_renyi','kraus_channel','UnsupportedLowering']
