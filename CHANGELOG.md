# Changelog

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
