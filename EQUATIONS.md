# Quantum-Equation Layers (v0.0.6)

Seven equations from quantum mechanics, each contributing a distinct *role* to a
learning pipeline. The layers are not competing score functions; they are
observable → dynamics → constraint → response → composition → preservation →
update, each usable on one tape without forcing a composite pipeline.

Each opcode below carries: its contract, the equation it realizes, the invariant
it exposes, and an exact classical lowering used as a derived control (a
same-function reference, not a hardware claim).

All new opcodes run in the opt-in `numpy_overlap` / `pennylane_overlap` dialect
with standard semantics. The legacy `numpy_semantic` / `pennylane_standard`
backends and all earlier programs are unchanged.

## I. Overlap — the observable

Observable content = the Gram matrix `G_ij = <psi_i|psi_j>`. Its phases,
spectrum and rank are what any downstream readout can and cannot see.

| opcode | contract | equation |
|---|---|---|
| `gram_spectrum` | gram → spectrum | eigenvalues of `G` (ascending, `eigvalsh`) |
| `spectral_participation` | spectrum → real | `(sum_k lambda_k)^2 / sum_k lambda_k^2` |
| `gram_coherence` | gram → real | `||G - diag(G)||_F` |
| `phase_ablate` | gram → gram | `G -> diag(G)` (drops all off-diagonal phase) |

Invariant / meaning: participation ratio is an effective rank (capacity);
`phase_ablate` destroying coherence isolates the magnitude-only control.

Classical lowering: all four are closed-form linear algebra; the tests recompute
the Gram by hand and compare eigenvalue spectra and norms.

## II. Schrödinger — the dynamics

Dynamics content = the time-ordered propagator `U = T exp(-i ∫ H dt)`. The
generator spectrum is a memory/horizon scale; non-commutativity is measured by
the commutator.

| opcode | contract | equation |
|---|---|---|
| `generator_spectrum` | hermitian → spectrum | `eigvalsh(H)` |
| `matrix_exponential` | hermitian, real → unitary | `U = exp(-i t H)` via `eigh` |
| `time_ordered_evolve` | hermitian `(...,K,D,D)`, real `(...,K)` → unitary | `U = P_{K-1} ... P_0`, `P_k = exp(-i t_k H_k)` |
| `commutator` | hermitian, hermitian → operator | `[A,B] = AB - BA` (anti-Hermitian) |
| `frobenius_norm` | operator → real | `sqrt(tr(M^dag M))` |

Invariant / meaning: `matrix_exponential` is unitary; reversed input order in
`time_ordered_evolve` gives a different propagator exactly when the generators do
not commute. `[A,B] = 0` is the classical limit of commuting observables.

Classical lowering: `exp(-i t H)` is compared against an independent `eigh`
computation; time-ordering is compared against the explicit matrix product.

## III. Duality — the constraint

Wave-particle duality is the constraint that visibility and which-path
information cannot both be saturated (Englert–Greenberger–Yasin). It is exposed
as an invariant, never as a semantic switch.

| opcode | contract | equation |
|---|---|---|
| `detector_duality` | ket, ket → real `(...,2)` | `V = |<d0|d1>|`, `D = sqrt(1 - V^2)` |
| `path_duality` | density (qubit) → real `(...,2)` | `P = |rho00 - rho11|`, `V = 2|rho01|` |
| `duality_slack` | real `(...,2)` → real | `1 - a^2 - b^2` |

Invariant: for pure states `D^2 + V^2 = 1` and `P^2 + V^2 = 1` exactly; for mixed
states `P^2 + V^2 <= 1` (dephasing shrinks `V` and leaves `P` fixed). The softmax
of attention is the extreme `D = 1, V = 0`.

Classical lowering: predictions are compared against the closed-form qubit
formulas; the relation is checked to machine precision.

## IV. Schwinger — the response

Source theory: one generating object (a Green function / propagator) encodes the
linear response to an external source. Only the free (Gaussian) sector is in the
VM; the full functional-derivative tower is a documented boundary.

| opcode | contract | equation |
|---|---|---|
| `resolvent` | hermitian, real → operator | `(H - omega)^{-1}` (spectrum must stay positive) |
| `proper_time_resolvent` | hermitian, real → operator | `(1 - e^{-(E-omega) L}) / (E-omega)` per eigenvalue |
| `source_response` | operator, amplitude → amplitude | `phi = G J` |
| `generating_functional` | operator, amplitude → real | `W[J] = 1/2 Re <J|G|J>` |
| `effective_action` | operator, amplitude → real | `Gamma = 1/2 Re <phi|G^-1|phi>` |

Invariant / meaning: `proper_time_resolvent` is the finite-cutoff value of
`(H-omega)^{-1} = ∫_0^inf ds e^{-(H-omega)s}`; its truncation error is bounded by
`e^{-(E_min-omega)L} / (E_min-omega)` with `L = NumericalPolicy.proper_time_cutoff`.
The gradient of `W[J]` reproduces the response `G J` exactly.

Classical lowering: the resolvent is checked by `(H - omega) G = I`; the
generating functional by the closed-form quadratic form.

Invariant / meaning (0.0.6 completion): `effective_action` is the Legendre
response of the free theory. For `phi(J) = G J` the identity
`Gamma(phi(J)) = W[J]` holds exactly, because
`1/2 <GJ|G^-1|GJ> = 1/2 <J|G|J>`. Still only the Gaussian sector: no
interacting `Gamma` tower, no `iε` retarded branch.

Classical lowering: `effective_action` is checked against an independent
`linalg.solve` of the kernel.

## V. Entanglement — the composition

Composition content = how parts combine into a whole and what the whole
forgets when a part is discarded. The Schmidt spectrum and the entanglement
entropy quantify exactly that; the swap test is the overlap of two parts.

| opcode | contract | equation |
|---|---|---|
| `tensor_product` | vectors, vectors → ket | `psi_A ⊗ psi_B` (batch-aware outer product) |
| `partial_trace` | density → density | `Tr_B rho_AB`; requires `dims`, retains the first factor |
| `schmidt_spectrum` | density → spectrum | `sqrt(eig(Tr_B rho))`, the Schmidt coefficients |
| `entanglement_entropy` | density → real | `-Tr lambda log2 lambda` of the reduced spectrum |
| `swap_test` | vectors, vectors → real | `|<a|b>|^2` |

Invariant / meaning: `partial_trace` preserves the trace; for a pure whole the
Schmidt coefficients squared sum to one; the entropy is 0 for product states
and `log2(d)` for maximally entangled pairs. `partial_trace` retains only the
first factor of `dims` (tracing out everything else); arbitrary subsystem
selection is a documented boundary.

Classical lowering: compared against explicit `kron`, an index-wise partial
trace, `linalg.svd` of the reshaped coefficient matrix, and the closed-form
squared overlap.

## VI. Symmetry — the preservation

Preservation content = what a generator cannot change. A symmetry generator
commutes with the dynamics; its irreducible eigenspaces are what the evolution
preserves, and the conservation defect measures commutativity.

| opcode | contract | equation |
|---|---|---|
| `symmetry_generator` | (none) → hermitian | Pauli string from required `label` over `ixyz` |
| `projector_to_irrep` | hermitian → hermitian | spectral projector onto eigenspace `eigenvalue` |
| `conserved_current` | ket, hermitian → real | `Re <psi|Q|psi>`, the conserved charge expectation |
| `conservation_defect` | hermitian, hermitian → real | `|[A, B]|_F` (zero iff commuting) |

Invariant / meaning: Pauli-string generators have spectrum `±1` exactly;
`projector_to_irrep` is idempotent and Hermitian, and the projectors over the
distinct eigenvalues resolve the identity. `conservation_defect` agrees
exactly with layer II's `commutator` + `frobenius_norm` (cross-layer
consistency, not a new law). Only Pauli-string labels over qubit wires are
supported; general group representations are a boundary.

Classical lowering: compared against `kron` Pauli products, `eigvalsh`
eigenprojectors, the closed-form expectation, and the Frobenius norm of the
explicit commutator.

## VII. Measurement — the update

Update content = what an observation does to the state. A Kraus instrument is
the general state update; a POVM is what it can see; postselection is the
conditioned branch (the counterfactual axis the VM is named after, now exposed
as standard algebra rather than a semantic law).

| opcode | contract | equation |
|---|---|---|
| `kraus_apply` | density, operator → density | `sum_k K_k rho K_k^dag` (batch-aware over the state) |
| `povm_probabilities` | density, hermitian → real | `Tr(rho E_k)` per effect |
| `postselect` | density, hermitian → density | `E rho E / Tr(rho E)` (Lüders update) |

Invariant / meaning: for a trace-preserving Kraus set the update keeps the
trace at one; a complete POVM sums to one; `postselect` returns a valid
density matrix and raises `NumericalDomainError` on a zero-probability effect
(undefined conditional state, never silently repaired). Kraus completeness is
*not* validated as an input contract: an incomplete set returns a trace below
one and that is the caller's declared model, not a VM repair.

Classical lowering: compared against explicit `sum_k K rho K^dag`, the
closed-form Born probabilities, and the hand-computed conditioned eigenstate.

## Coverage boundaries (kept explicit)

- No `iε` retarded branch is implemented; the resolvent requires a positive
  shift (`E_min > omega`).
- No functional derivatives / higher correlators: `generating_functional` is the
  free Gaussian `W[J]` only; `effective_action` is its exact Legendre partner,
  still Gaussian. No interacting tower.
- Duality is a probed invariant, not a new `SemanticProfile` axis.
- `partial_trace` retains the first factor only; no arbitrary subsystem
  selection, no `partial_transpose`, no negative-eigenvalue entanglement
  witnesses.
- `symmetry_generator` accepts Pauli strings over `ixyz` only; no general
  group labels, irreducible-character tables, or Noether currents on
  continuous configuration spaces.
- `kraus_apply` performs the algebraic channel but does not validate Kraus
  completeness; `postselect` conditions on a single effect, not a general
  instrument with classical registers.
- The PennyLane backend shares the same host algebra for post-encoding opcodes;
  it independently prepares states, not every downstream op.