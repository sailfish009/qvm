# Quantum-Equation Layers (v0.0.5)

Four equations from quantum mechanics, each contributing a distinct *role* to a
learning pipeline. The layers are not four competing score functions; they are
observable → dynamics → constraint → response on one tape.

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

Invariant / meaning: `proper_time_resolvent` is the finite-cutoff value of
`(H-omega)^{-1} = ∫_0^inf ds e^{-(H-omega)s}`; its truncation error is bounded by
`e^{-(E_min-omega)L} / (E_min-omega)` with `L = NumericalPolicy.proper_time_cutoff`.
The gradient of `W[J]` reproduces the response `G J` exactly.

Classical lowering: the resolvent is checked by `(H - omega) G = I`; the
generating functional by the closed-form quadratic form.

## Coverage boundaries (kept explicit)

- No `iε` retarded branch is implemented; the resolvent requires a positive
  shift (`E_min > omega`).
- No functional derivatives / higher correlators: `generating_functional` is the
  free Gaussian `W[J]` only.
- Duality is a probed invariant, not a new `SemanticProfile` axis.
- The PennyLane backend shares the same host algebra for post-encoding opcodes;
  it independently prepares states, not every downstream op.