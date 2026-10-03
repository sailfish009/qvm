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
