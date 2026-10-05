"""Run one small program per quantum-equation layer and print its invariant.

This is a diagnostics example, not a benchmark. It shows that the seven layers
share one tape and that each exposes its defining quantity.
"""
import numpy as np
from qvm import (
    QVM, NumericalPolicy,
    overlap_participation, overlap_coherence,
    generator_evolution, generator_spectrum, commutator_strength,
    time_ordered_sequence,
    detector_duality_program, path_duality_program, duality_slack_program,
    resolvent_program, source_response_program, generating_functional_program,
    effective_action_program, entanglement_entropy_program, swap_test_program,
    symmetry_generator_program, irrep_projector_program, conservation_defect_program,
    instrument_channel_program, povm_probability_program, postselect_program,
)


def main():
    vm = QVM()
    backend = 'numpy_overlap'

    # I. overlap: effective rank and phase coherence of a Gram matrix.
    angles = np.array([[0.0], [0.4], [1.2], [2.1]])
    participation = float(vm.run(overlap_participation(), {'angles': angles},
                                 backend=backend))
    coherence = float(vm.run(overlap_coherence(), {'angles': angles}, backend=backend))
    print(f'I   overlap        participation={participation:.6f} coherence={coherence:.6f}')

    # II. Schrodinger: generator spectrum, propagator unitarity, non-commutativity.
    generator = np.array([[1.0, 0.2], [0.2, -0.5]])
    unitary = np.asarray(vm.run(generator_evolution(),
                                {'generator': generator, 'time': 0.7}, backend=backend))
    unitarity = np.max(np.abs(unitary.conj().T @ unitary - np.eye(2)))
    spectrum = np.asarray(vm.run(generator_spectrum(), {'generator': generator},
                                 backend=backend))
    strength = float(vm.run(commutator_strength(),
                            {'left': np.diag([1.0, 2.0]), 'right': np.array([[0.0, 1.0], [1.0, 0.0]])},
                            backend=backend))
    generators = np.stack([np.diag([1.0, 2.0]), np.array([[0.0, 1.0], [1.0, 0.0]])])
    times = np.array([0.6, 0.9])
    ordered = np.asarray(vm.run(time_ordered_sequence(),
                                {'generators': generators, 'times': times}, backend=backend))
    reversed_ = np.asarray(vm.run(time_ordered_sequence(),
                                  {'generators': generators[::-1], 'times': times[::-1]},
                                  backend=backend))
    print(f'II  schrodinger    spectrum={np.round(spectrum, 6).tolist()} '
          f'unitarity={unitarity:.2e} commutator={strength:.6f} '
          f'order_split={np.linalg.norm(ordered - reversed_):.6f}')

    # III. duality: D^2 + V^2 = 1 for pure detectors, and dephasing shrinks V.
    pair = np.asarray(vm.run(detector_duality_program(),
                             {'left': np.array([1.0, 0.0]), 'right': np.array([0.0, 1.0])},
                             backend=backend))
    theta = 0.8
    state = np.array([np.cos(theta / 2), np.sin(theta / 2)])
    rho = np.outer(state, state.conj())
    pure = np.asarray(vm.run(path_duality_program(), {'state': rho}, backend=backend))
    mixed = np.asarray(vm.run(path_duality_program(), {'state': np.diag(np.diag(rho))},
                              backend=backend))
    slack = float(vm.run(duality_slack_program(), {'pair': pure}, backend=backend))
    print(f'III duality        detector={np.round(pair, 6).tolist()} '
          f'pure_PV={np.round(pure, 6).tolist()} mixed_V={mixed[1]:.6f} slack={slack:.2e}')

    # IV. Schwinger: resolvent, finite proper-time control, and response.
    omega = -1.0
    propagator = np.asarray(vm.run(resolvent_program(),
                                   {'generator': generator, 'source_energy': omega},
                                   backend=backend))
    approximate = np.asarray(vm.run(resolvent_program(truncated=True),
                                    {'generator': generator, 'source_energy': omega},
                                    backend=backend))
    cutoff = NumericalPolicy().proper_time_cutoff
    smallest = float(np.min(np.linalg.eigvalsh(generator) - omega))
    bound = np.exp(-smallest * cutoff) / smallest
    source = np.array([1.0, 0.5])
    field = np.asarray(vm.run(source_response_program(),
                              {'propagator': propagator, 'source': source}, backend=backend))
    action = float(vm.run(generating_functional_program(),
                          {'propagator': propagator, 'source': source}, backend=backend))
    print(f'IV  schwinger      resolvent_ok={np.max(np.abs((generator - omega * np.eye(2)) @ propagator - np.eye(2))):.2e} '
          f'proper_time_err={np.max(np.abs(approximate - propagator)):.2e} '
          f'(<= {bound:.2e}) action={action:.6f} response={np.round(field, 6).tolist()}')

    # V. entanglement: entropy of a Bell pair, swap fidelity of random parts.
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    bell_rho = np.outer(bell, bell.conj())
    entropy = float(vm.run(entanglement_entropy_program((2, 2)), {'state': bell_rho},
                           backend=backend))
    part = np.array([np.cos(0.3), np.sin(0.3)])
    other = np.array([np.cos(1.1), np.sin(1.1)])
    fidelity = float(vm.run(swap_test_program(), {'left': part, 'right': other},
                            backend=backend))
    print(f'V   entanglement   bell_entropy={entropy:.6f} swap_fidelity={fidelity:.6f}')

    # VI. symmetry: charge spectrum is +-1, projectors resolve the identity.
    charge = np.asarray(vm.run(symmetry_generator_program('zz'), {}, backend=backend))
    split = np.max(np.abs(np.abs(np.linalg.eigvalsh(charge)) - 1))
    generator_sym = np.array([[0.0, 1.0], [1.0, 0.0]])
    total = np.zeros((2, 2))
    for eigenvalue in (1.0, -1.0):
        total += np.asarray(vm.run(irrep_projector_program(eigenvalue),
                                   {'generator': generator_sym}, backend=backend))
    defect = float(vm.run(conservation_defect_program(),
                          {'left': np.diag([1.0, 2.0]), 'right': np.array([[0.0, 1.0], [1.0, 0.0]])},
                          backend=backend))
    print(f'VI  symmetry       charge_spectrum_ok={split:.2e} '
          f'projectors_sum_to_I={np.max(np.abs(total - np.eye(2))):.2e} defect={defect:.6f}')

    # VII. measurement: dephasing channel and a conditioned postselection.
    ket = np.array([np.cos(0.4), np.sin(0.4)])
    rho = np.outer(ket, ket.conj())
    dephase = np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
    updated = np.asarray(vm.run(instrument_channel_program(),
                                {'state': rho, 'kraus': dephase}, backend=backend))
    effects = dephase
    probabilities = np.asarray(vm.run(povm_probability_program(),
                                      {'state': rho, 'effects': effects}, backend=backend))
    collapsed = np.asarray(vm.run(postselect_program(),
                                  {'state': rho, 'effect': np.diag([1.0, 0.0])},
                                  backend=backend))
    print(f'VII measurement    dephased_offdiag={np.max(np.abs(updated - np.diag(np.diag(updated)))):.2e} '
          f'povm_sum={float(np.sum(probabilities)):.6f} '
          f'postselect_|0><0|_ok={np.max(np.abs(collapsed - np.diag([1.0, 0.0]))):.2e}')


if __name__ == '__main__':
    main()