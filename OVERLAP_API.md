# Overlap dialect (0.0.6)

Select `backend="numpy_overlap"` or `backend="pennylane_overlap"` explicitly. The default backend remains
`numpy_semantic` for legacy compatibility. New opcodes are not silently inserted into old executors. The seven
equation layers (I overlap, II Schrodinger, III duality, IV Schwinger, V entanglement, VI symmetry, VII measurement)
are documented in `EQUATIONS.md`.

## Value types and layout

Array arguments should be NumPy/PennyLane NumPy arrays. Leading axes are batch axes. Symbols:
`D` = Hilbert dimension, `N` = collection size, `W` = number of wires.

| Type | Layout / contract |
|---|---|
| `ket` | `(..., D)`, finite, unit squared norm on last axis |
| `amplitude` | `(..., D)`, finite; zero and non-unit vectors allowed |
| `state_collection` | `(..., N, D)`, each last-axis vector normalized |
| `density` | `(..., D, D)`, Hermitian PSD, trace one |
| `gram` | `(..., N, N)`, Hermitian PSD; no trace-one or unit-diagonal requirement on external Gram inputs |
| `operator` | `(..., D, D)`, arbitrary finite square matrix; reserved general value contract |
| `hermitian` | `(..., D, D)`, Hermitian |
| `unitary` | `(..., D, D)`, U†U = I |
| `angles` | real finite array, detailed shape depends on encoder |
| `coefficients` | `(..., N)`, real or complex; no probability constraint |
| `complex` | finite scalar or tensor, real dtype also allowed |
| `real` | finite real-dtype scalar or tensor |
| `spectrum` | `(..., D)`, finite real eigenvalues (ascending where produced by `eigvalsh`) |

Instruction `role` describes intent; `result_type` describes the value. A ket is a representation of a pure density
state, not an alternative physical state space. A Gram matrix or amplitude is a mathematical intermediate, not a
normalized density state. Inputs use `p.input(name, qtype, shape=None)`; outputs use
`p.emit(..., role="host_math", result_type="...")`.

Every overlap intermediate is checked, including during Autograd tracing using primal values. Validation does not
modify or detach the executed expression. The types of outputs are inferred from operator contracts when omitted;
explicit annotations must agree. In contrast, legacy programs keep their original opcode-specific checks plus
return-value validation; this is not a full typed rewrite of the legacy interpreter.

## Operators

| Opcode | Inputs → output | Attributes / meaning |
|---|---|---|
| `encode_ry` | angles → ket | `(..., W)`; product of RY states |
| `encode_ryrz` | angles → ket | `(..., W, 2)`; RY then RZ on each wire |
| `as_collection` | ket → state_collection | require an existing item axis; reinterpret, do not add an axis |
| `inner_product` | vectors, vectors → complex | sum conjugate(left) * right over last axis |
| `gram_matrix` | state_collection → gram | G_ij = <s_i\|s_j> |
| `absolute_square` | complex or gram → real | elementwise conjugate(z)*z; not matrix multiplication |
| `linear_combination` | state_collection, coefficients → amplitude | sum_n c_n s_n; no conjugation of coefficients |
| `normalize` | amplitude → ket | max-rescaled exact normalization; zero raises |
| `ket_density` | ket → density | ket times its adjoint |
| `apply_unitary` | ket, unitary → ket | U psi, standard semantics only |
| `expectation` | ket, hermitian → real | real <psi\|O\|psi>, exact host expectation |
| `cyclic_overlap` | gram → complex | required `indices=[i,j,k,...]`; product around closed cycle |
| `ridge_project` | state_collection, ket → amplitude | required finite `ridge > 0`; regularized span smoother |

For `inner_product`, accepted vectors are ket, amplitude, or state_collection, with broadcast-compatible leading
axes and equal D. To get all pairs use `gram_matrix` for one collection, or explicitly supply broadcast axes for
two collections. Automatic query/key roles are deliberately absent.

`operator` is a validation type, not a license for an arbitrary nonunitary physical evolution. Only the opcodes
listed above and the equation-layer opcodes in `EQUATIONS.md` are supported. `tensor_product` composes two pure
state vectors, `kraus_apply` applies a declared operator set, and `postselect` conditions on one effect; these are
algebraic contracts, not arbitrary program gates. Scans, general projector construction, arbitrary subsystem
partial traces, and pseudoinverses are not part of the dialect. Some density/tensor operations exist separately in
the legacy executor. Add operations in response to concrete experiments, not as an assumed model architecture.

## Regularized projection and singularity

For a matrix S with states as rows, define G_ij = <s_i|s_j> and z_i = <s_i|x>. Then

```text
c = solve(G + ridge * I, z)
y = sum_i c_i s_i
```

This is defined for rank-deficient collections with explicit positive ridge, subject to finite-precision
conditioning. It is not the pseudoinverse projector, is not generally idempotent, and can depend on duplicated
states. Rephasing each state with the matching Gram/overlap transformation preserves the output. The trace records
`declared_ridge_regularization`; no hidden eigenvalue floor or automatic ridge is added.

## Autograd and PennyLane

Both overlap backends support real trainable arrays and real scalar losses through complex state algebra.
For example:

```python
import pennylane as qml
from pennylane import numpy as np
from qvm import QVM, encoded_overlap
vm = QVM()
p = encoded_overlap(complex_encoding=True, fidelity=True)
right = np.array([[0.7, 0.2]], requires_grad=False)
def loss(theta):
    return vm.run(p, {"left": theta, "right": right}, backend="pennylane_overlap")
theta = np.array([[0.3, 0.5]], requires_grad=True)
print(qml.grad(loss)(theta))
```

`pennylane_overlap` uses cached `default.qubit` QNodes with `interface="autograd"` and `diff_method="backprop"` for
state preparation. Batches execute one state per QNode invocation to avoid wide-state broadcast limitations.
All remaining opcodes are shared explicit host algebra, including `apply_unitary` and `expectation`; they are not
silently claimed to be independent physical circuit lowerings. Trace entries distinguish `pennylane_qnode` from
`host_algebra`. Use the legacy backend's supported circuits when that independent density/channel comparison is
wanted. The legacy backend remains verification-only, with no new training guarantee.

The validator rejects unsupported ops, extra attributes, incompatible types, shapes, and nonstandard profiles.
There is no name-based replacement tape. Standard profiles with custom validation tolerances are accepted by the
new dialect; those tolerances are recorded in the audit's numerical policy.

## Phase conventions and interpretation

RY/RZ order fixes the encoder phase convention. Raw overlap changes under independent global state rephasing.
Fidelity and cyclic products are invariant; a coherent sum must transform coefficients inversely if rephasing is
only a change of representation. Changing actual relative preparation phases can change readout. These are
separate interventions.

An unnormalized sum can have squared norm above one. Do not call that norm a physical success probability without
an explicitly contractive filter and reference implementation. Similarly, regularization and algebraic projection
are not quantum gates or evidence of quantum advantage.
