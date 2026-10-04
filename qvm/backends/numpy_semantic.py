"""Dense differentiable executor parameterized by explicit semantics.

This is the legacy counterfactual backend: the same tape can be re-executed
under different SemanticProfiles, and every deviation from standard quantum
mechanics is recorded in the trace.
"""
import numpy as np
from pennylane import numpy as anp

from ..numerics import UndefinedBranch, ordinary

I2 = anp.eye(2, dtype=complex)
X = anp.array([[0, 1], [1, 0]], dtype=complex)
Y = anp.array([[0, -1j], [1j, 0]], dtype=complex)
Z = anp.array([[1, 0], [0, -1]], dtype=complex)
SWAP = anp.array([[1, 0, 0, 0], [0, 0, 1, 0],
                  [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
COMPUTATIONAL_PROJECTORS = anp.stack((anp.array([[1, 0], [0, 0]], dtype=complex),
                       anp.array([[0, 0], [0, 1]], dtype=complex)))


def density(bloch):
    return (I2 + bloch[0] * X + bloch[1] * Y + bloch[2] * Z) / 2


class NumpySemanticBackend:
    name = 'numpy_semantic'

    def execute(self, program, inputs, semantics, return_trace=False):
        env = {}
        trace = []
        evolution_event = 0
        policy = semantics.numerical_policy
        for index, ins in enumerate(program.instructions):
            op = ins.op
            args = [env[name] for name in ins.inputs]
            used = []
            interventions = []
            event = None

            if op == 'input':
                value = inputs[ins.output]
            elif op == 'density_from_bloch':
                value = density(args[0])
                semantics.validate_state(value)
                used = ['state_space']
            elif op == 'ry_product_state':
                value = anp.array([1. + 0j])
                for theta in args[0]:
                    value = anp.kron(value, anp.array([anp.cos(theta / 2),
                                                       anp.sin(theta / 2)], dtype=complex))
                used = ['state_space', 'composition_rule']
            elif op == 'statevector_density':
                value = anp.outer(args[0], anp.conj(args[0]))
                used = ['state_space']
            elif op == 'pure_fidelity':
                amplitude = anp.vdot(args[0], args[1])
                value = anp.real(amplitude * anp.conj(amplitude))
            elif op == 'tensor':
                value = semantics.compose(args[0], args[1], interventions)
                used = ['composition_rule']
            elif op == 'unitary':
                value = semantics.unitary(args[0], args[1], interventions)
                used = ['evolution_spectral_power', 'evolution_application_scope',
                        'numerical_policy']
                event = evolution_event
                evolution_event += 1
            elif op == 'partial_swap':
                angle = args[1]
                operator = anp.cos(angle) * anp.eye(4) - 1j * anp.sin(angle) * SWAP
                value = semantics.unitary(args[0], operator, interventions)
                used = ['evolution_spectral_power', 'evolution_application_scope',
                        'numerical_policy']
                event = evolution_event
                evolution_event += 1
            elif op == 'partial_trace_second':
                reshaped = anp.reshape(args[0], (2, 2, 2, 2))
                value = reshaped[:, 0, :, 0] + reshaped[:, 1, :, 1]
            elif op == 'dephase':
                value = semantics.channel(args[0], COMPUTATIONAL_PROJECTORS, interventions)
                used = ['evolution_spectral_power', 'evolution_application_scope',
                        'numerical_policy']
                event = evolution_event
                evolution_event += 1
            elif op == 'kraus_channel':
                value = semantics.channel(args[0], args[1], interventions)
                used = ['evolution_spectral_power', 'evolution_application_scope',
                        'numerical_policy']
                event = evolution_event
                evolution_event += 1
            elif op == 'povm_probabilities':
                value = semantics.probabilities(args[0], args[1], interventions)
                used = ['measurement_exponent', 'numerical_policy']
            elif op == 'projective_measurement':
                value = semantics.selective_projective_measurement(
                    args[0], args[1], interventions)
                used = ['measurement_exponent', 'collapse_strength', 'numerical_policy']
            elif op == 'measurement_probabilities':
                value = args[0].probabilities
            elif op == 'measurement_branches':
                value = args[0].branches
            elif op == 'branch_state':
                outcome = int(ins.attrs['outcome'])
                defined = args[0].defined[outcome]
                if ordinary(defined) and not bool(np.asarray(defined)):
                    raise UndefinedBranch(
                        f'outcome {outcome} has zero Born probability; its conditional state is undefined')
                value = args[0].branches[outcome]
            elif op == 'uhlmann_fidelity':
                left = policy.sqrt(args[0], interventions, 'uhlmann_left_sqrt')
                middle = policy.sqrt(left @ args[1] @ left, interventions, 'uhlmann_middle_sqrt')
                value = anp.real(anp.trace(middle)) ** 2
                used = ['numerical_policy']
            elif op == 'stabilize_fidelity':
                eps = float(ins.attrs['eps'])
                value = (args[0] + eps) / (1 + eps)
                interventions.append({'kind': 'declared_fidelity_stabilization', 'epsilon': eps})
                used = ['numerical_policy']
            elif op == 'bures_squared':
                value = 2 * (1 - anp.sqrt(anp.maximum(args[0], 0)))
                used = ['numerical_policy']
            elif op == 'sandwiched_renyi':
                value = self._sandwiched_renyi(args, ins.attrs, policy, interventions)
                used = ['numerical_policy']
            elif op == 'pauli_readout':
                value = anp.stack(tuple(anp.real(anp.trace(args[0] @ p)) for p in (X, Y, Z)))
            else:
                raise NotImplementedError(f'{self.name}: {op}')

            env[ins.output] = value
            entry = {
                'index': index,
                'output': ins.output,
                'op': op,
                'role': ins.role,
                'semantics': semantics.name,
                'assumptions_used': used,
                'numerical_interventions': interventions,
            }
            if event is not None:
                entry['semantic_event_index'] = event
            trace.append(entry)
        result = env[program.output]
        return (result, trace) if return_trace else result

    @staticmethod
    def _sandwiched_renyi(args, attrs, policy, interventions):
        alpha = float(attrs['alpha'])
        if abs(alpha - 1) < 1e-12:
            rho, sigma = args
            return anp.real(anp.trace(
                rho @ (policy.log(rho, interventions, 'relative_entropy_log_rho')
                       - policy.log(sigma, interventions, 'relative_entropy_log_sigma'))))
        power = (1 - alpha) / (2 * alpha)
        left = policy.power(args[1], power, interventions, 'renyi_sigma_power')
        middle = left @ args[0] @ left
        sandwiched = policy.power(middle, alpha, interventions, 'renyi_sandwiched_power')
        return anp.log(anp.real(anp.trace(sandwiched))) / (alpha - 1)