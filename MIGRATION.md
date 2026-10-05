# Migrating 0.0.5 → 0.0.6

## Preserved

- Distribution `qvm-counterfactual`, import `qvm`, `QVM.run(...)`, default backend and all legacy backend names.
- Every program, `SemanticProfile` field, value type and numerical-policy field from 0.0.5. No new value types
  are introduced: reduced states reuse `density`, Schmidt coefficients reuse `spectrum`, charge generators and
  POVM effects reuse `hermitian`, Kraus operators reuse `operator`.
- All 59 retained tests, including the schema round-trips, now resealing to `qvm_v0.0006`.

## Changed

1. New serializations use `qvm_v0.0006`; schemas v0.0002–v0.0005 load and reseal as v0.0006. Identity hashes
   change across schema versions; keep original JSON/hashes as provenance.
2. The opt-in overlap dialect adds the thirteen opcodes in `EQUATIONS.md` (layers V/VI/VII and the `effective_action`
   completion of IV). They require standard semantics.
3. Three opcodes carry required attributes: `dims` (tuple of positive ints, first factor retained),
   `label` (Pauli string over `ixyz`) and `eigenvalue` (finite real).
4. `tensor_product` and `kraus_apply` are batch-aware over leading axes; earlier v0.0006 development snapshots
   used `kron`/`matmul` contractions that were wrong for batched inputs and a `partial_trace` contraction that
   was mathematically incorrect even unbatched. Both are fixed and regression-tested before release.

## Not promised

The new layers are small standard-algebra primitives, not a claim of quantum advantage or a new counterfactual
law. `partial_trace` retains the first factor only; `symmetry_generator` accepts Pauli strings only; `kraus_apply`
does not validate Kraus completeness; `postselect` conditions on a single effect and rejects zero-probability
branches. The interacting `Gamma` tower and the `iε` retarded branch remain out of scope.

```bash
python -m pip install -e .
python -m unittest discover -s tests
```

# Migrating 0.0.4 → 0.0.5

## Preserved

- Distribution `qvm-counterfactual`, import `qvm`, `QVM.run(...)`, default backend and all legacy backend names.
- Every legacy program (`product_ry_fidelity`, `mixed_bures`, `sandwiched_renyi`, `partial_swap`, `kraus_channel`),
  every `SemanticProfile` field, and every numerical-policy field from 0.0.4 (new fields have defaults).
- All 42 retained tests, including the schema round-trip, now resealing to `qvm_v0.0005`.

## Changed

1. New serializations use `qvm_v0.0005`; schemas v0.0002/v0.0003/v0.0004 load and reseal as v0.0005.
   Identity hashes change across schema versions; keep original JSON/hashes as provenance.
2. `NumericalPolicy` adds `proper_time_cutoff` (default 40.0) and `hermitian_roundoff_tolerance` (default 1e-10).
   The former parameterizes finite proper-time integration; the latter makes a non-Hermitian matrix an explicit
   `NumericalDomainError` instead of being silently symmetrized.
3. The value-type set adds `spectrum`. Generators reuse `hermitian`, sources reuse `amplitude`, propagators/commutators
   reuse `operator`.
4. The opt-in overlap dialect adds the sixteen opcodes in `EQUATIONS.md`. They require standard semantics and are the
   only place the new value type is produced.
5. `tools/build_upload.py` derives the version and archive tag from `qvm.__version__` instead of a hardcoded string.

## Not promised

The new layer opcodes are small standard-algebra primitives, not a claim of quantum advantage or a new
counterfactual law. `resolvent` requires a positive spectral shift; only the free Gaussian generating functional is
implemented; duality is a probed invariant, not a new semantic axis.

```bash
python -m pip install -e .
python -c "import qvm; print(qvm.__version__)"
python -m unittest discover -s tests -v
```

# Migrating 0.0.3 → 0.0.4

## Preserved

- Distribution `qvm-counterfactual`, import `qvm`, `QVM.run(...)`, legacy backend names, and default backend.
- Legacy semantic profile fields and numerical-policy manifests.
- Historical product-RY fidelity and other standard equation formulas.
- Counterfactual profiles and diagnostic property reports. They are not candidate-selection gates.
- Backend registration remains available; existing custom backends are not automatically converted to the new dialect.

## Changed

1. `Program.emit` accepts optional `result_type=...`; `Instruction` stores it separately from `attrs`.
2. New serializations use `qvm_v0.0004`. Schemas v0.0002/v0.0003 load and reseal as v0.0004.
   Identity hashes therefore change across schema versions; preserve original JSON/hashes as provenance.
   Unannotated legacy tapes retain the same structural records. Adding type annotations changes structure.
3. Returning `ry_product_state` now validates a ket, not a density matrix. Legacy output type inference is based on
   known opcodes; unknown custom state outputs should explicitly declare their `result_type`.
4. New value validation uses explicit absolute tolerance (`rtol=0`). Valid standard reference cases reproduce;
   malformed values accepted only through a loose relative tolerance may now fail.
5. Sealed programs reject assignment/deletion of internal fields as well as the public API. Attributes must be
   finite JSON-compatible values, not mutable arrays or arbitrary objects.
6. `numpy_overlap` and `pennylane_overlap` are new opt-in backends. They require standard semantics and validate a
   composable dialect, not one hardcoded entire program fingerprint.

## Not promised

The old `pennylane_standard` backend is still a verification backend for three canonical tapes. Installing 0.0.4
does not make it generally differentiable or make legacy instructions legal in the new overlap dialect.

The previously reproduced 99.74% attention model is a historical reference, not a required architecture. This
release checks the preserved 12-wire fidelity equation but does not claim a new full attention retraining run.
New examples need neither QKV nor attention.

## Update

```bash
python -m pip install -e .
python -c "import qvm; print(qvm.__version__, qvm.__file__)"
python -m unittest discover -s tests -v
```

Do not mutate archived experiment environments silently: record the imported version and source path in future
runs. Saved parameter arrays alone are not complete experiment provenance.
