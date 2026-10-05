# Assumption and numerical-policy model

## Principle

A backend answers an immutable program under a semantic profile. It does not define that profile. Agreement across
software backends tests implementation consistency under shared assumptions, not the truth of those assumptions.

## Typed overlap extension (0.0.4), extended by the equation layers (0.0.5, 0.0.6)

The new overlap dialect supports standard complex Hilbert algebra without an attention/QKV architecture. A ket is
another representation of a pure density state; amplitudes, Gram matrices, coefficients and operators are distinct
mathematical values. They must not all be validated as trace-one density matrices. Types do not introduce a new
physical law. The overlap backends reject counterfactual profiles rather than ignoring changed assumptions.

`normalize` rejects the zero vector and uses max-rescaled normalization without an epsilon fallback. `ridge_project`
requires and audits a positive ridge; its output is a regularized span smoother, not an exact projector or gate.
Raw overlap phases depend on preparation conventions; cyclic products and fidelities have different invariances.
Arbitrary coherent-sum norms are not physical success probabilities. See `OVERLAP_API.md`.

The 0.0.6 layers are algebraic contracts, not physical claims. `partial_trace` retains the first declared factor
and what it returns is a mathematical reduced state, not a claim about spatial subsystems. Entanglement entropy is
a property of the declared `dims` factorization, not an observable-independent number. `symmetry_generator` Pauli
strings are a finite label set, not a representation-theory engine. `kraus_apply` does not validate Kraus
completeness: a subnormalized output is the caller's declared model, not a VM repair. `postselect` conditions on a
single effect and rejects zero-probability branches as undefined, consistent with the counterfactual discipline of
never silently repairing an undefined conditional state.

## Standard profile

The standard profile declares complex positive-semidefinite trace-one states, tensor-product composition, linear
unitary/CPTP evolution, Born probabilities, Lüders selective update and convex input mixtures. These are explicit
configuration, not unquestioned VM laws.

## Counterfactual axes

- Escort measurement: `P_alpha(i) = p_i^(alpha/2) / sum_j p_j^(alpha/2)`, alpha>0.
- Partial selective update: `(1-kappa)rho + kappa rho_i`, kappa in[0,1].
- Spectral-power evolution: `rho^beta / Tr(rho^beta)`, beta>0, after each declared evolution instruction.

Alpha2, kappa1 and beta1 are the standard endpoints. Nonstandard profiles are controlled mathematical models,
not proposed laws of nature.

## Numerical policy is separate

`NumericalPolicy` is recorded independently from physical semantics.

- Positive matrix powers preserve exact zero eigenvalues; no positive mass is injected into null support.
- Negative eigenvalues within the declared float64 roundoff tolerance are projected to zero and every occurrence
  is recorded as a numerical intervention.
- Matrix logarithms and negative powers on singular support raise `NumericalDomainError`; this implementation does not hide
  those domains behind an eigenvalue floor.
- Measurement powers use max-rescaled normalization, preventing equal nonzero weights from jointly underflowing
  at large finite exponents.
- Tiny negative probability residue may be projected to zero only within the declared tolerance and is audited.
- NaN and infinite semantic parameters or ordinary inputs are rejected.

A deliberately regularized matrix function must be represented by an explicit program operation or future named
policy. It must not masquerade as the exact semantic law.

## Conditional branches

Every strictly positive Born-probability branch is divided by its actual probability, however small, and therefore
has trace one. A zero-probability conditional branch is mathematically undefined. Full measurement results expose
a `defined` mask; direct access raises `UndefinedBranch`. A zero matrix may occupy the unused array slot but is
never presented as a valid conditional state.

## Evolution events and program decomposition

The implemented application scope is `per_evolution_instruction`. Each evolution opcode is a separate semantic
event and is numbered in execution traces. Consequently a beta!=1 profile can distinguish one identity evolution
from two identity evolutions. This is an explicit property of the profile, not a compiler-invariant assumption.
Alternative event grouping is not implemented in this release.

## Declared versus used assumptions

Audit results distinguish:

- the complete declared semantic manifest;
- assumptions actually invoked by executed opcodes;
- changed assumptions unused by the program;
- numerical interventions actually applied.

For example, an escort exponent is declared but unused by a Bures-only host-math program.

## Properties and counterexamples

`qvm.properties` checks enforced invariants separately from exploratory properties. Counterexamples are retained as
artifacts rather than silently rejected. The bundled suite covers outcome refinement, repeated measurement,
convex-mixture affinity, identity-evolution decomposition and tensor associativity. A counterexample characterizes
the selected profile; it does not by itself prove or disprove a physical theory.

## Coverage boundary

Held fixed in the legacy semantic profiles are complex density states, tensor composition and convex input mixtures. Alternative scalar
fields, state spaces, composition laws, signed states, hidden-variable models, hardware behavior and unknown
alternatives are not covered. Positive learning utility would not establish physical truth, and a negative result
would exclude only the tested profiles and tasks.
