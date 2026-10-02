# Contributing

1. State which assumption an operation changes and which assumptions remain fixed.
2. Add a canonical semantic manifest field and document known departures.
3. Implement dense reference semantics before optimized or hardware lowering.
4. Never silently lower a counterfactual operation or an unmatched instruction tape to a standard circuit.
5. Keep physical semantics separate from `NumericalPolicy`; record every numerical repair actually applied.
6. Add rank-deficient, tiny-probability, extreme-parameter, finite-difference, serialization and invariant tests.
7. Declare assumptions actually used by each opcode, not only the complete selected profile.
8. Preserve property violations as reproducible counterexamples rather than filtering them out.
9. Report both implemented and unexamined alternatives.
10. Do not describe a useful learning operation as a law of nature without independent physical evidence.

Run:

```bash
python -m unittest discover -s tests -v
python examples/reproduce_reliability_cases.py
python examples/reproduce_standard_semantics.py
python examples/assumption_sweep.py
python examples/discover_counterexamples.py
```
