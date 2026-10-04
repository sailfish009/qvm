# Changelog

## 0.0.5

Four quantum-equation concept layers in one opt-in typed dialect, plus a
minimalist core refactor. Standard semantics are unchanged; no new
counterfactual law and no superiority claim over attention.

- I. overlap (observable): `gram_spectrum`, `spectral_participation`,
  `gram_coherence`, `phase_ablate`;
- II. Schrodinger (dynamics): `generator_spectrum`, `matrix_exponential`,
  `time_ordered_evolve`, `commutator`, `frobenius_norm`;
- III. duality (constraint): `detector_duality`, `path_duality`, `duality_slack`
  expose the D^2+V^2<=1 complementarity invariant;
- IV. Schwinger (response): `resolvent`, `proper_time_resolvent`,
  `source_response`, `generating_functional` (stationary-action response);
- added the `spectrum` value type; generators reuse `hermitian`, sources reuse
  `amplitude`, Green functions reuse `operator`;
- added `NumericalPolicy.proper_time_cutoff` for finite proper-time integration;
- schema is now `qvm_v0.0005`; v2/v3/v4 JSON imports are resealed into v5;
- refactored `ir.py`, `numerics.py`, `semantics.py`, `vm.py` and the dense
  backend back to a readable, minimal style (one statement per line, no dead
  code) and made matrix Hermiticity checks explicit instead of silent;
- made Bloch-input validation batch-aware;
- `tools/build_upload.py` now reads the version from `qvm.__version__`;
- added `EQUATIONS.md` and `tests/test_equations_v5.py` with closed-form
  classical lowerings for every new opcode;
- retained all v4 tests and behavior (59 tests total).

## 0.0.4

Typed overlap-learning infrastructure, with no attention/QKV requirement:

- added explicit result types independent of instruction roles;
- fixed pure-state return validation, including legacy product-RY states;
- added complex/batched overlap, Gram, cyclic products, superposition, normalization, density conversion,
  pure-state unitary/readout, and explicit ridge span smoothing;
- added a validated composable standard-complex dialect, not program-name dispatch;
- added differentiable NumPy and actual PennyLane-QNode state encoders, with shared host algebra explicitly labeled;
- added a no-attention overlap-metric training example and phase/Gram geometry diagnostics;
- strengthened internal sealing and finite JSON attribute contracts;
- imported v2/v3 serialized programs into schema v4; legacy formulas and default executor retained;
- retained previous reliability regressions and added typed/batch/gradient/phase/no-PyTorch tests;
- documented numerical domains, physical interpretation limits, backend scope, and migration;
- did not add a new counterfactual law or claim superiority over attention.

## 0.0.3

Reliability correction with no new counterfactual semantic axis:

- separated `NumericalPolicy` from semantic-law parameters;
- preserved exact zero support for positive matrix powers;
- made singular logarithms and negative powers explicit domain errors;
- added stable max-rescaled measurement powers and finite-parameter checks;
- normalized every positive-probability branch exactly and marked zero-probability branches undefined;
- deeply sealed program instructions, attributes, names and returns;
- changed PennyLane dispatch from program names to exact structural tape matching;
- recorded declared, used and unused changed assumptions plus actual numerical interventions;
- declared spectral application scope as one application per evolution instruction;
- added deterministic property probes and preserved counterexample artifacts;
- added regression tests for all independently reported edge cases;
- retained v0.0002 JSON import with resealing into the v0.0003 schema.

## 0.0.2

- Separated Program, SemanticProfile and Backend.
- Added Born/escort measurement, partial-collapse and spectral-power evolution axes.
- Added standard-only PennyLane rejection for nonstandard profiles.
- Added standard-source reproduction and assumption sweeps.
- Added packaging and public-release materials.
