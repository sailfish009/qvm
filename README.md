# qvm_v0.0002

**A small counterfactual-semantics virtual machine for quantum learning equations.**

Most quantum simulators ask: *what does a chosen quantum model predict?* This project asks an earlier question:
*which assumptions produced that prediction, and what changes when one assumption is replaced?*

`Program`, `SemanticProfile`, and `Backend` are independent inputs:

```text
result = QVM.run(program, inputs, semantics, backend)
```

The standard complex-density/Born/Lüders/linear-CPTP model is included as one explicit profile. It is not hidden
inside the VM and is not treated as empirically proven by simulator agreement.

> This project is not Rigetti's QVM, does not execute Quil, and is not a hardware emulator. The simple directory
> name is retained for this research lineage. It makes no claim that its counterfactual profiles are laws of
> nature.

## Why

Agreement between NumPy and PennyLane can show implementation consistency while both implementations share the
same physical assumptions. qvm_v0.0002 records and replaces those assumptions so that conclusions remain
conditional on a visible coverage set.

Implemented assumption axes:

| Axis | Standard point | Counterfactual family |
|---|---|---|
| Measurement | Born exponent2 | Escort exponent alpha>0 |
| Selective update | Lüders collapse strength1 | Partial/no collapse kappa in[0,1] |
| Evolution | Linear map, spectral power1 | Normalized spectral power beta>0 after evolution |

Held standard in this release: complex density states, tensor-product composition, and convex input mixtures.
Alternative scalar fields, alternative composition laws, signed states and unknown alternatives are explicitly
**not covered**.

## Install

After downloading or cloning the repository, open a terminal in its root directory. For active research, use an
editable install so source changes are immediately visible:

```bash
python -m pip install -e .
```

Then import it from any working directory:

```python
import qvm
print(qvm.__version__)
```

For a frozen installation, install the built wheel instead:

```bash
python -m pip install dist/qvm_counterfactual-0.0.2-py3-none-any.whl
```

A wheel is normally attached to a GitHub Release rather than committed with the source tree.

The distribution name is `qvm-counterfactual`; the Python import name is `qvm`. Check an installation with
`python -m pip show qvm-counterfactual`. Because `qvm` is a generic import name, use an isolated virtual
environment if another installed project exposes the same namespace. The validated environment uses Python3.14,
NumPy2.4.6 and PennyLane0.45.1. PyTorch is not required.

## Quick start

```python
import numpy as np
from qvm import QVM, povm_measurement, standard_profile, escort_profile

I = np.eye(2, dtype=complex)
Z = np.diag([1, -1]).astype(complex)
effects = np.array([(I + 0.8*Z)/2, (I - 0.8*Z)/2])
inputs = {"r": np.array([0.3, 0.1, 0.4]), "effects": effects}

vm = QVM()
program = povm_measurement()

standard = vm.run(program, inputs, standard_profile(), audit=True)
alternative = vm.run(program, inputs, escort_profile(1.4), audit=True)

assert standard.program_sha256 == alternative.program_sha256
print(standard.value, alternative.value)
print(alternative.semantics["changed_assumptions"])
```

The program hash is unchanged; only the semantic profile changes.

## Profiles

### Standard

```python
standard_profile()
```

Declares complex density states, tensor products, linear CPTP evolution, Born probabilities and Lüders selective
updates.

### Escort measurement

For raw Born probabilities `p_i`,

```text
P_alpha(i) = p_i^(alpha/2) / sum_j p_j^(alpha/2)
```

`alpha=2` is the standard point. Other values are counterfactual measurement semantics, not physical claims.

```python
escort_profile(alpha=1.5)
```

### Partial collapse

For the Lüders branch state `rho_i`,

```text
rho_i(kappa) = (1-kappa) rho + kappa rho_i
```

`kappa=1` is Lüders; `kappa=0` leaves the state unchanged.

```python
partial_collapse_profile(kappa=0.5)
```

### Spectral-power evolution

After an otherwise standard evolution,

```text
rho -> rho^beta / Tr(rho^beta)
```

`beta=1` is linear standard evolution. Other values are nonlinear and generally do not preserve convex mixtures.

```python
spectral_power_profile(beta=1.7)
```

Profiles can be combined with `counterfactual_profile(...)`. Every profile produces a canonical assumption
manifest, a hash, enforced invariants, and known departures from the standard profile.

## Programs and semantic roles

Instructions are tagged as `data`, `state`, `composition`, `evolution`, `measurement`, `host_math`, or `output`.
Programs are sealed after construction and serialize to canonical JSON with SHA-256 identity.

Included programs:

- POVM and selective projective measurement;
- unitary and Kraus evolution;
- product-RY fidelity;
- mixed Bures geometry;
- sandwiched Rényi geometry;
- tensor/partial-SWAP/partial trace.

Host geometry such as Bures or matrix logarithms is not mislabeled as a physical gate.

## Backends

- `numpy_semantic`: differentiable dense execution under any implemented profile.
- `pennylane_standard`: independent circuit execution under the standard profile only.

A nonstandard profile sent to `pennylane_standard` raises `UnsupportedLowering`. The VM never silently converts a
counterfactual semantic law into a standard PennyLane circuit.

## Reproduce

```bash
python -m unittest discover -s tests -v
python examples/reproduce_standard_semantics.py
python examples/assumption_sweep.py
```

The standard profile reproduces preserved product-fidelity, Bures, Rényi and partial-SWAP equations. The sweep
shows that one unchanged program produces distinct normalized outputs under declared rival assumptions.

See `RESULTS.md`, `ASSUMPTIONS.md`, and the JSON artifacts under `artifacts/`.

## Interpretation boundary

A profile yielding better machine-learning performance would show that its mathematical operation is useful. It
would not by itself show that nature follows that profile. Conversely, agreement of standard NumPy and
PennyLane backends establishes consistency under shared assumptions, not proof of those assumptions.

No finite set of profiles covers every way an assumption can fail. Every report must list implemented, held
fixed, and unexamined assumptions. This limitation is a design requirement, not a footnote.

## License

Apache-2.0. Review attribution and repository metadata before publishing under a personal or organizational
GitHub account.
