# Frozen standard-reference vectors

`standard_vectors.npz` contains100 deterministic input/output cases generated from the preserved research
implementations listed in `MANIFEST.json`. It allows the public repository to reproduce the standard semantic
endpoint without requiring unrelated sibling repositories.

The manifest records the RNG seed, historical numerical parameters, source SHA-256 values and NPZ hash. These
vectors verify lineage only; they do not establish that the standard semantic assumptions are physically true.
