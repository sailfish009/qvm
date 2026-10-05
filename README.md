# QVM 0.0.6 — quantum-equation layers without an attention requirement

**Small, auditable quantum-equation programs for NumPy and PennyLane.**

QVM is execution infrastructure, not a proposed physical theory, a model architecture, or a quantum-advantage claim.
Version 0.0.6 adds three more composable **quantum-equation concept layers** to the 0.0.5 tape — entanglement
(composition), symmetry (preservation) and measurement (update) — and completes the free Schwinger response with
its exact Legendre partner, with no Q/K/V, softmax, or Transformer. Earlier counterfactual-semantics and overlap
examples remain available, but are not a gate for learning research.

```text
QVM.run(program, inputs, semantics, backend)
```

Program, semantic assumptions, numerical policy, and execution backend remain separate. This project is not
Rigetti's QVM and does not execute Quil.

## What's new

- Seven equation layers sharing one opt-in typed dialect; see `EQUATIONS.md` for contracts, math and invariants:
  - I. overlap: `gram_spectrum`, `spectral_participation`, `gram_coherence`, `phase_ablate`.
  - II. Schrodinger: `generator_spectrum`, `matrix_exponential`, `time_ordered_evolve`, `commutator`, `frobenius_norm`.
  - III. duality: `detector_duality`, `path_duality`, `duality_slack` (the `D^2 + V^2 <= 1` invariant).
  - IV. Schwinger: `resolvent`, `proper_time_resolvent`, `source_response`, `generating_functional`,
    `effective_action` (the exact Legendre partner `Gamma(phi(J)) = W[J]`).
  - V. entanglement: `tensor_product`, `partial_trace`, `schmidt_spectrum`, `entanglement_entropy`, `swap_test`.
  - VI. symmetry: `symmetry_generator`, `projector_to_irrep`, `conserved_current`, `conservation_defect`.
  - VII. measurement: `kraus_apply`, `povm_probabilities`, `postselect` (Lüders update, zero-probability branches
    raise instead of being repaired).
- No new value types: reduced states reuse `density`, Schmidt coefficients reuse `spectrum`, charge generators and
  POVM effects reuse `hermitian`, Kraus operators reuse `operator`.
- `tensor_product` and `kraus_apply` are batch-aware over leading axes.
- Added the `spectrum` value type in 0.0.5; explicit result types, independent of instruction roles: ket, amplitude,
  state collection, density, Gram, operator, Hermitian operator, unitary, complex, real, angles, coefficients, and
  spectrum.
- Fixed returning a valid pure state: it is no longer incorrectly validated as a square density matrix (0.0.4).
- Batched real/complex state encoders, complex overlaps, Gram matrices, cyclic overlap products, linear
  superpositions, explicit normalization, pure-state evolution, density conversion, and expectation readout.
- Explicitly regularized span projection with a reported ridge, not a disguised exact projector.
- Differentiable `numpy_overlap` and `pennylane_overlap` backends for a documented instruction subset.
- An attention-free overlap-metric training example, with PennyLane QNodes in the actual gradient path.
- Old standard-equation and reliability regressions retained; old program JSON accepted (v2–v5 reseal as v6).

## Install

Python 3.11 or newer; NumPy >=2; PennyLane 0.45.x. No PyTorch, GPU, QPU, network dataset, or sibling research
repository is required.

```bash
python -m pip install -e .
python -c "import qvm; print(qvm.__version__)"
```

Distribution: `qvm-counterfactual`; import: `qvm`; version: `0.0.6`. Use a virtual environment if another package
uses the generic `qvm` namespace. A release wheel can be installed instead of the editable source.

## Raw complex overlap, not necessarily fidelity

```python
import numpy as np
from qvm import QVM, encoded_overlap

vm = QVM()
program = encoded_overlap(complex_encoding=True)
inputs = {
    # (..., wires, 2): RY angle, then RZ angle on each wire
    "left": np.array([[0.4, 0.7], [0.8, -0.2]]),
    "right": np.array([[1.0, -0.1], [0.3, 0.6]]),
}
a = vm.run(program, inputs, backend="numpy_overlap", audit=True)
b = vm.run(program, inputs, backend="pennylane_overlap")
np.testing.assert_allclose(a.value, b, atol=1e-14)
print(a.value)                         # <left|right>, generally complex
print(a.trace)
```

Set `fidelity=True` only when squared magnitude is the desired operation. The unsquared amplitude has a phase
convention fixed by the preparation circuit. It is not by itself invariant to independent global rephasings of
its two states.

## Build a program without QKV

```python
from qvm import Program

p = Program("set_geometry")
p.input("angles", "angles")
p.emit("states", "encode_ryrz", "angles", role="state", result_type="ket")
p.emit("collection", "as_collection", "states", result_type="state_collection")
p.emit("G", "gram_matrix", "collection", result_type="gram")
p.emit("cycle", "cyclic_overlap", "G", indices=[0, 1, 2], result_type="complex")
p.returns("cycle")
```

For angles shaped `(items, wires, 2)`, this returns `G[0,1] G[1,2] G[2,0]`, invariant under each state's independent
global rephasing. Leading batch axes are supported. A Gram matrix is **not** a density matrix: no trace-one
constraint or automatic trace normalization is applied to it.

See [OVERLAP_API.md](OVERLAP_API.md) for the full supported dialect and [MIGRATION.md](MIGRATION.md) for compatibility.

## Backends and what they verify

| Backend | Scope | Differentiation / verification |
|---|---|---|
| `numpy_overlap` | New standard-complex overlap dialect | PennyLane NumPy / Autograd, explicit batched algebra |
| `pennylane_overlap` | Same validated composable dialect | RY/RZ state preparation through actual differentiable QNodes; subsequent host algebra is shared |
| `numpy_semantic` | Legacy density/measurement/evolution programs | Existing Autograd paths and counterfactual profiles; not a universal differentiability promise |
| `pennylane_standard` | Three legacy canonical tapes | Independent circuit verification; **not** a general training backend |

The new dialect rejects counterfactual profiles, unknown opcodes, unknown attributes, wrong arity, and incompatible
types. It interprets every supported instruction, including a changed return value; names do not select a hidden
circuit. The old PennyLane backend still uses exact structural matching of its supported canonical tapes.

PennyLane overlap agreement independently checks state preparation, not every downstream matrix operation.
Separate direct-algebra, invariance, and finite-difference tests check those host operations. Returning a simulator
statevector is not free amplitude access on a QPU.

## Run

```bash
python -m unittest discover -s tests -v
python examples/inspect_overlap_geometry.py
python examples/train_overlap_embedding.py
python examples/reproduce_standard_semantics.py
python examples/reproduce_reliability_cases.py
python examples/assumption_sweep.py
python examples/discover_counterexamples.py
```

The learning example trains one shared input-to-state encoder against pairwise similarity targets. It has no
query/key/value projections or attention. Both backends train from the same parameters and data; histories,
checkpoints, held-out similarities, and derivative checks are saved under `artifacts/`.

This is a small engineering example, **not** evidence that overlap beats attention or another learner. Larger
comparative experiments remain separate from this package release. There are no candidate-promotion thresholds.

## Numerical and physical boundaries

- Zero amplitude may be represented but cannot be normalized; this raises `NumericalDomainError`.
- Normalization rescales by the maximum absolute component before computing its norm. No epsilon state is added.
- `ridge_project` requires an explicit positive ridge, records it, and is not an idempotent orthogonal projector.
- Ket, density, and Gram validation have different mathematical contracts. Validation does not repair inputs.
- State norms of arbitrary coherent sums are **not** postselection success probabilities. A physical filter
  requires a specified contractive scaling and implementation.
- Finite differences are checked for real trainable parameters and real losses through complex intermediates.
  Nonsmooth points, singular inverse problems, arbitrary eigendecomposition derivatives, and second derivatives
  are not generally guaranteed.
- There is no generic QPU compiler, finite-shot estimator, sparse/tensor-network backend, or broad gate-language
  interpreter in this release. Dense state preparation remains exponential in wire count.

Counterfactual assumptions and numerical policy remain documented in [ASSUMPTIONS.md](ASSUMPTIONS.md). Property
counterexamples describe the selected mathematical profile; they are not learning-selection gates.

## Publication

See [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md) for GitHub's **Add file → Upload files** workflow.
Local verification is not hosted CI verification. Upload and release creation are performed by the repository owner.

## License

Apache-2.0; see `LICENSE`. Preserve attribution when redistributing.
