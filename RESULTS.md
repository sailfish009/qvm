# qvm_v0.0002 results

## Decision

The replaceable-semantics VM is implemented and passes its infrastructure acceptance tests. One serialized
program can now run under standard or counterfactual measurement, collapse and evolution assumptions without
changing its program hash.

This is not evidence that a counterfactual profile describes nature. It is evidence that assumptions are no
longer inseparable from the executable program in this environment.

## Implemented semantic axes

| Axis | Standard | Alternatives |
|---|---|---|
| Measurement | Born exponent2 | Escort exponent alpha>0 |
| Selective update | Lüders strength1 | Partial/no collapse in[0,1] |
| Evolution | Linear, beta1 | Normalized spectral power beta>0 |

Every execution audit contains the program hash, semantic manifest/hash, changed assumptions, invariants,
known nonstandard properties, backend, instruction trace and input hashes.

## Standard-profile lineage

One hundred random cases were compared with preserved research source files.

| Equation | Maximum absolute error |
|---|---:|
| product-RY fidelity |1.46e-16|
| historical mixed Bures |1.64e-14|
| sandwiched Rényi alpha=.9 |1.60e-13|
| partial-SWAP density update |2.22e-16|

These are conditional reproductions under the declared standard profile.

## Counterfactual sweep

For1,000 random qubit states and binary POVMs, mean total-variation distance from standard Born output was:

| Measurement exponent | Mean TV | Maximum TV |
|---:|---:|---:|
|1.0|.0333|.1497|
|1.5|.0162|.0640|
|2.0|0|0|
|3.0|.0296|.0897|
|4.0|.0555|.1501|

Every output remained normalized to numerical precision. Alpha2 exactly reproduced the standard profile.

Partial-collapse sweeps preserved the same outcome probabilities while continuously changing branch states from
no update at kappa0 to Lüders update at kappa1. Spectral-power sweeps preserved PSD and trace while changing
purity; beta1 exactly reproduced linear standard evolution.

These observations only establish that the semantic axes are active and isolated. They are not learning results
or physical anomalies.

## Tests

Eleven tests pass:

- program serialization, sealing and semantic independence;
- standard/escort endpoint identity and normalization;
- trainable measurement-exponent gradient versus finite difference;
- partial-collapse endpoint and interpolation identities;
- spectral-power standard endpoint and purity change;
- preserved standard equations and PennyLane partial-SWAP;
- refusal to lower counterfactual semantics to standard PennyLane circuits;
- explicit assumption manifests and hashes;
- state/POVM validation;
- semantic-role classification;
- absence of PyTorch.

## Coverage boundary

Implemented alternatives do not cover every possible failure of standard assumptions. This release holds complex
density states, tensor products and convex input mixtures fixed. It does not implement alternative scalar fields,
composition laws, signed states, hidden-variable models, nonlocal alternatives or unknown unknowns.

Therefore the VM supports conditional falsification and comparison; it cannot certify that all possible semantic
alternatives were considered.

## Publication status

The directory contains packaging metadata, Apache-2.0 text attributed to `sailfish009`, README, assumption
documentation, contribution rules, tests, portable frozen references, examples and reproducibility artifacts. It
has not been pushed to GitHub. A clean file-upload bundle is provided; the repository name/description and hosted
CI result remain user-side publication steps.
