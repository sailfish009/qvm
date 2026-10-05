# 0.0.6 local verification results

## Scope

This release adds three quantum-equation concept layers (entanglement, symmetry, measurement) to the 0.0.5 tape
and completes the free Schwinger response with `effective_action`. Standard semantics are unchanged. It is an
infrastructure release, not a comparative learning experiment and not a claim of quantum advantage.

Validated locally with Python 3.14, NumPy 2.4.6 and PennyLane 0.45.1. The hosted CI matrix must still be run after upload.

## Tests and equation checks

- **88 unit tests passed**: 59 retained 0.0.5 regressions and 29 new layer regressions.
- Every new opcode is checked against an independent closed-form reference (its exact classical lowering).
- `partial_trace` preserves the trace; the Bell pair reduces to `I/2`; product states stay pure; dimension
  mismatches raise.
- Schmidt coefficients match `linalg.svd` of the reshaped coefficient matrix and their squares sum to one.
- Entanglement entropy extremes: 0 for product states, 1 for the Bell pair.
- `tensor_product` matches `kron` row-wise for batched inputs (a batch-unsafe `kron` lowering was found and
  replaced before release).
- `symmetry_generator` Pauli strings have spectrum exactly `±1`; irrep projectors are idempotent, Hermitian and
  resolve the identity over the distinct eigenvalues.
- `conservation_defect` agrees with layer II's `commutator` + `frobenius_norm` to machine precision.
- A trace-preserving Kraus set keeps the trace at one; a complete POVM sums to one; `postselect` reproduces the
  hand-computed conditioned eigenstate and rejects zero-probability effects.
- The Legendre identity `Gamma(phi(J)) = W[J]` holds for the free resolvent theory.
- The `swap_test` gradient flows through amplitude inputs and matches the closed-form derivative.

Reference checks (max over random batches):

| Check | Maximum error |
|---|---:|
| `partial_trace` of the Bell pair vs `I/2` | 1.1e-16 |
| `schmidt_spectrum` vs `linalg.svd` | 1.1e-16 |
| `tensor_product` vs `kron` (batched) | 0 |
| `swap_test` vs squared overlap | 0 |
| `symmetry_generator` vs `kron` Pauli product | 0 |
| irrep projectors vs eigenprojectors and idempotence | 1.1e-16 |
| `conservation_defect` vs `commutator` strength | 0 |
| `kraus_apply` vs explicit `sum_k K rho K^dag` | 1.1e-16 |
| `povm_probabilities` vs closed-form Born rule | 0 |
| `postselect` vs conditioned eigenstate | 1.1e-16 |
| `effective_action` vs `linalg.solve` | 2.2e-16 |
| Legendre identity `Gamma(phi(J)) = W[J]` | 1.1e-16 |

## Remaining limitations

- `partial_trace` retains the first factor of `dims` only; no `partial_transpose` witnesses.
- `symmetry_generator` accepts Pauli strings over `ixyz` only; no general representation theory.
- `kraus_apply` does not validate Kraus completeness; an incomplete set is the caller's declared model.
- `effective_action` is the Gaussian Legendre partner only; no interacting tower, no `iε` retarded branch.
- No finite-shot, hardware, speedup or physical-law claim follows from simulator state access.

# 0.0.5 local verification results

## Scope

This release implements four quantum-equation concept layers (overlap, Schrodinger, duality, Schwinger) as an
opt-in typed dialect plus a minimalist core refactor. Standard semantics are unchanged. It is an infrastructure
release, not a comparative learning experiment and not a claim of quantum advantage.

Validated locally with Python 3.14, NumPy 2.4.6 and PennyLane 0.45.1. The hosted CI matrix must still be run after upload.

## Tests and equation checks

- **59 unit tests passed**: 42 retained legacy/overlap regressions and 17 new four-layer regressions.
- Every new opcode is checked against an independent closed-form reference (its exact classical lowering).
- Non-commutativity is demonstrated by reversing `time_ordered_evolve` inputs and asserting a changed propagator.
- Duality `D^2 + V^2 = 1` holds for random pure detectors; dephasing shrinks path visibility and preserves predictability.
- `proper_time_resolvent` converges to `resolvent` and its error is asserted below the analytic truncation bound.
- The `generating_functional` gradient reproduces the response `G J` exactly.

Reference checks (max over random batches):

| Check | Maximum error |
|---|---:|
| `matrix_exponential` vs independent `eigh` and unitarity | 2.89e-15 |
| `time_ordered_evolve` vs explicit ordered product | 0 |
| `resolvent` vs `(H - omega) G = I` | 9.33e-15 |
| detector duality `D^2 + V^2 = 1` | 2.22e-16 |

`proper_time_resolvent` differs from `resolvent` by `exp(-(E_min-omega) L)/(E_min-omega)` up to the finite cutoff
`L`. The discrepancy is small when the source energy is well below the spectrum and grows near threshold, exactly as
the truncation formula predicts. This is reported, not repaired; the cutoff is a declared numerical policy.

## Minimalist refactor

`ir.py`, `numerics.py`, `semantics.py`, `vm.py` and `backends/numpy_semantic.py` were rewritten in a readable,
minimal style (one statement per line, no dead code). The dead `unitary` branch in `vm._validate` was removed; matrix
Hermiticity is now checked explicitly; Bloch-input validation is batch-aware. Behavior is unchanged, verified by the
42 retained tests.

## Remaining limitations

- New opcodes are deliberately small standard-algebra primitives, not arbitrary quantum programs.
- No retarded `iε` branch and no functional-derivative tower; only the free Gaussian generating functional.
- Duality is an exposed invariant, not a new `SemanticProfile` axis.
- No finite-shot, hardware, speedup or physical-law claim follows from simulator state access.
- New scientific experiments comparing these layers with attention/classical architectures remain future work.

# 0.0.4 local verification results

## Scope

This release implements QKV-free typed overlap algebra and differentiable execution. It is an infrastructure
release, not a comparative learning experiment or a claim that attention is harmful. The historical 99.74% model
remains a reference case rather than a mandatory backbone.

Validated locally with Python 3.14, NumPy 2.4.6 and PennyLane 0.45.1. The hosted Python 3.11–3.14 CI matrix must still
be run after the owner uploads the source.

## Tests and equation checks

- **42 unit tests passed**: 22 retained legacy tests and 20 overlap/type/gradient regressions.
- Returning an unannotated legacy `ry_product_state` now correctly returns a normalized ket.
- Typed intermediate validation, complex conjugation, Gram PSD, independent rephasing, cyclic products,
  coherent-sum norm identities, cancellation, and rank-deficient ridge cases are checked.
- Normalization accepts representable tiny/huge nonzero amplitudes and explicitly rejects zero; no epsilon state
  or artificial success branch is inserted.
- Autograd versus finite differences is tested through complex encoding, Gram/ridge solves, coefficient learning,
  normalization, readout, density conversion, unitary evolution and cyclic products.
- A subprocess blocks every PyTorch import and trains the small example through both overlap backends.
- Unknown ops/attributes, incompatible result annotations and nonstandard overlap profiles are rejected.
- Supported changes to return values execute the actual tape rather than a circuit chosen by program name.

Across 100 random batched complex-overlap cases:

| Check | Maximum error |
|---|---:|
| NumPy preparation vs PennyLane QNode overlap | 3.61e-16 |
| coherent norm versus c†Gc (example) | 0 |
| cyclic product under independent state rephasing (example) | 2.22e-17 |
| regularized span output under state rephasing (example) | 2.69e-16 |

PennyLane independently prepares states; subsequent algebra is shared host math and separately checked against
explicit algebraic references. These are implementation/invariance checks, not proof of quantum advantage.

## Attention-free learning smoke example

`examples/train_overlap_embedding.py` trains a shared 12-parameter input-to-state encoder with pairwise fidelity
loss. There are no Q/K/V projections, Transformer, or softmax routing. The second run uses actual differentiable
PennyLane QNodes during optimization.

Same data, initialization, full batches, learning rate 0.3, 40 steps, seed 404:

| Metric | NumPy overlap | PennyLane overlap |
|---|---:|---:|
| Initial training pair loss | 0.479322 | 0.479322 |
| Final training pair loss | 0.002937 | 0.002937 |
| Held-out pair loss | 0.011856 | 0.011856 |

- Final parameter difference: 5.55e-17.
- Held-out similarity difference: 8.88e-16.
- PennyLane full-loss directional finite-difference error: 8.35e-11.
- Histories, data, parameters and predictions: `artifacts/overlap_learning.json` and `.npz`, generated by the example.

This demonstrates a functioning gradient path, not a benchmark, statistical confirmation, or superiority over
attention/classical learning. No promotion criteria are applied.

## Legacy reproduction

100 frozen standard cases still reproduce:

| Equation | Maximum error |
|---|---:|
| product-RY fidelity | 1.46e-16 |
| mixed Bures | 1.65e-14 |
| sandwiched Renyi (alpha 0.9) | 1.60e-13 |
| partial-SWAP | 2.23e-16 |

The prior reliability cases and four expected diagnostic counterexamples remain covered. The 12-wire historical
fidelity equation is checked, but a new full 99.74% attention retraining run is not part of this release.

## Remaining limitations

- New overlap backends support a deliberately small standard-complex dialect, not arbitrary quantum programs.
- Legacy PennyLane lowering remains verification-only.
- No finite-shot, hardware cost, quantum speedup or physical-law claim follows from simulator state access.
- A coherent-sum norm is not automatically a postselection probability.
- The explicit ridge smoother is neither an exact projector nor invariant to arbitrary changes in sampling or
  duplicated basis vectors.
- New scientific experiments are still needed to compare overlap-based representation, set geometry and dynamics
  with attention or other architectures. This release makes those experiments less architecture-constrained.
