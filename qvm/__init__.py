"""qvm_v0.0003: auditable counterfactual semantic execution for quantum learning equations."""
from .ir import Program,Instruction
from .vm import QVM,ExecutionResult
from .numerics import NumericalPolicy,NumericalDomainError,UndefinedBranch
from .semantics import SemanticProfile,SelectiveMeasurementResult,standard_profile,escort_profile,partial_collapse_profile,spectral_power_profile,counterfactual_profile
from .programs import product_ry_fidelity,povm_measurement,projective_measurement,unitary_evolution,partial_swap,mixed_bures,sandwiched_renyi,kraus_channel
from .backends.pennylane_standard import UnsupportedLowering
from .properties import PropertyResult,inspect_profile,default_property_report
__version__='0.0.3'
__all__=['PropertyResult','inspect_profile','default_property_report','Program','Instruction','QVM','ExecutionResult','NumericalPolicy','NumericalDomainError','UndefinedBranch','SemanticProfile','SelectiveMeasurementResult','standard_profile','escort_profile','partial_collapse_profile','spectral_power_profile','counterfactual_profile','product_ry_fidelity','povm_measurement','projective_measurement','unitary_evolution','partial_swap','mixed_bures','sandwiched_renyi','kraus_channel','UnsupportedLowering']
