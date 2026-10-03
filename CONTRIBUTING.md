# Contributing

QVM supports experiments; it must not impose attention, QKV, or a physical-law search as the research objective.

1. Start with a concrete equation and the experiment that needs it. State mathematical domains and representation.
2. Keep immutable Program, semantics, numerical policy, and backend separate.
3. Distinguish result types from instruction roles: ket, amplitude, density, and Gram have different contracts.
4. Implement small composable ops with explicit type, shape, attribute, and backend support. Do not hide a model in
   a name-dispatched backend or accept attributes that are ignored.
5. Declare which assumptions are used. Reject unsupported semantics instead of silently running standard math.
6. Add direct-algebra references, adversarial tapes, batch tests, and finite differences of real losses through
   complex intermediates. Distinguish shared host algebra from independently checked PennyLane circuits.
7. Test zero amplitudes, rank-deficient collections, conjugation, independent rephasing, and numerical extremes.
8. Do not hide regularization, clipping, or fallback states; record intended regularizers and actual repairs.
9. Preserve negative learning results and diagnostic property counterexamples. Do not use them as automatic
   promotion/rejection gates or claim that simulator agreement establishes physical truth.
10. Keep public examples self-contained. No machine-specific paths, credentials, proprietary datasets, or PyTorch.

```bash
python -m unittest discover -s tests -v
python examples/inspect_overlap_geometry.py
python examples/train_overlap_embedding.py
python examples/reproduce_reliability_cases.py
python examples/reproduce_standard_semantics.py
python examples/assumption_sweep.py
python examples/discover_counterexamples.py
```

See `OVERLAP_API.md` for the intentionally limited current scope, and `MIGRATION.md` before changing schemas.
