# qvm_v0.0003 reliability results

## Scope

Version0.0.3 is a reliability correction to the counterfactual-semantics VM. It adds no new semantic axis and makes
no physical-law or quantum-advantage claim. Program, SemanticProfile and Backend remain independent. The goal is
to distinguish law changes from numerical repair, invalid conditional branches and backend dispatch behavior.

## Independently reported regressions

All reported cases were first reproduced against0.0.2 and then frozen as0.0.3 tests.

| case | 0.0.2 behavior | 0.0.3 behavior |
|---|---|---|
| `diag(1,0)`, spectral beta=.1 | `diag(.9617135,.0382865)` | exactly `diag(1,0)` |
| positive branch probability `5e-15` | branch trace `.5` | branch trace `1` |
| zero-probability direct branch | numeric placeholder accessible | raises `UndefinedBranch` |
| equal probabilities, alpha=10000 | `[NaN,NaN]` | exactly `[.5,.5]` |
| NaN/Inf semantic parameters | accepted until invalid output | rejected before execution |
| spoofed `partial_swap*` program name | PennyLane ran a different tape | `UnsupportedLowering` |
| sealed instruction attrs/return | mutable | public mutations rejected |

Positive powers now preserve null support. Singular logarithms and negative powers use explicit strict-domain errors
rather than an eigenvalue floor. Roundoff projections are governed by a separate `NumericalPolicy` and included in
the execution audit when they actually occur.

## Backend and program integrity

PennyLane lowering no longer dispatches on `program.name`. It requires an exact name-independent structural hash
of a supported opcode sequence, attributes and return value. Renaming an otherwise identical canonical program is
accepted; changing its returned intermediate value is rejected. Nonstandard semantic profiles remain rejected.

Sealing converts the instruction tape to immutable tuples and recursively freezes instruction attributes. Program
name and output are read-only properties. Mutating a detached `record()` cannot alter the program. v0.0002 JSON
can be read, after which it is resealed and serialized under the v0.0003 schema with a new identity hash.

## Declared versus used assumptions

Audit records now include `used_assumptions`, `unused_changed_assumptions` and `numerical_interventions`. For an
escort-alpha4 profile executing a Bures-only program, `born_exponent` is correctly reported as declared but unused.
The same profile executing POVM measurement reports `measurement_exponent` as used.

The spectral application scope is explicitly `per_evolution_instruction`; trace entries number each semantic
evolution event. No instruction-decomposition invariance is silently assumed.

## Property and counterexample report

The deterministic property explorer evaluated28 profile/property cells. All enforced probability-simplex,
density-state and tensor-associativity checks passed. Four expected exploratory counterexamples were retained:

| profile | property | maximum difference |
|---|---|---:|
| escort alpha4 | outcome refinement/recombination | .1666667 |
| partial collapse kappa=.5 | selective repeatability | .2 |
| spectral beta2 | one versus two identity-evolution events | .0549325 |
| spectral beta2 | convex-mixture affinity | .1274808 |

These are characteristics of the declared alternatives, not discarded samples or evidence that the alternatives
are laws of nature.

## Regression and source reproduction

Twenty-two tests pass, covering the original API, trainable semantic gradients, exact endpoints, branch domains,
extreme exponents, deep sealing, structural lowering, used-assumption audit and counterexample preservation.
PyTorch is not imported.

The portable100-case standard reference remains reproduced under the declared standard profile:

| equation | maximum error |
|---|---:|
| product-RY fidelity |1.46e-16|
| historical mixed Bures |1.64e-14|
| sandwiched Rényi alpha=.9 |1.60e-13|
| partial-SWAP density update |2.22e-16|

These are implementation-consistency results conditional on the profile. They do not validate the underlying
physical assumptions.

## Coverage boundary

Implemented alternatives remain escort measurement, partial collapse and post-evolution spectral power. Held
fixed are complex density states, tensor composition and convex input mixtures. Alternative scalar fields, state
spaces, composition rules, signed states, hidden variables, physical hardware behavior and unknown alternatives
remain uncovered.
