# Contributing

1. State which assumption an operation changes and which assumptions remain fixed.
2. Add a canonical semantic manifest field and document known departures.
3. Implement dense reference semantics before optimized or hardware lowering.
4. Never silently lower a counterfactual operation to a standard circuit.
5. Add boundary, finite-difference, serialization and invariant tests.
6. Report both implemented and unexamined alternatives.
7. Do not describe a useful learning operation as a law of nature without independent physical evidence.

Run:

```bash
python -m unittest discover -s tests -v
python examples/reproduce_standard_semantics.py
python examples/assumption_sweep.py
```
