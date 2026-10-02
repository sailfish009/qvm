# Assumption model

## Principle

A backend answers a program under a semantic profile. It does not define the profile. Re-running one profile on
multiple software backends tests implementations, not the profile's truth.

## Standard profile

The bundled standard profile assumes:

1. States are positive semidefinite complex matrices with trace one.
2. Composite states use the Kronecker tensor product.
3. Closed evolution is linear and unitary.
4. Declared channels are linear CPTP maps.
5. POVM probabilities are `Tr(rho E_i)`.
6. Selective projective measurement uses Lüders update.
7. Input mixtures are convex and linear.

These assumptions are explicit configuration, not QVM axioms.

## Counterfactual profiles

### Escort measurement

Given a valid POVM and standard raw values `p_i=Tr(rho E_i)`, the profile returns

`P_alpha(i)=p_i^(alpha/2)/sum_j p_j^(alpha/2)`.

It preserves nonnegativity and normalization for alpha>0. Except at alpha=2 it does not assert the standard Born
rule or its usual noncontextual interpretation. The implementation does not claim consistency with every
possible composite-system or no-signalling experiment.

### Partial selective update

After outcome i, it returns `(1-kappa)rho+kappa rho_i^Luders`. This remains a valid density matrix for kappa in
[0,1], but for kappa<1 repeat measurement need not have the standard repeatability property. Outcome
probabilities and update law are deliberately separable assumptions.

### Spectral-power evolution

After a declared evolution it applies `rho^beta/Tr(rho^beta)`. It preserves Hermiticity, positivity and trace,
but beta!=1 is nonlinear in rho and generally violates convex-mixture preservation. It is a controlled
counterfactual map, not a proposed fundamental equation.

## Invariants versus assumptions

This release keeps PSD and trace-one state validity for every profile. Therefore it cannot test whether positivity
or density matrices themselves are wrong. A future signed-state profile must declare different validators rather
than bypassing current validation silently.

## Coverage rule

Every experiment must publish three lists:

- implemented alternatives;
- assumptions held fixed;
- alternatives not covered.

A negative result excludes only the tested profiles on the tested tasks. A positive learning result establishes
utility of an operation, not physical truth. A physical claim requires independent observations and controls.

## Numerical policy

Matrix powers floor eigenvalues below1e-14. Measurement probabilities with tiny negative floating-point residue
are clamped to zero before normalization. Both policies are in every profile manifest. They are numerical
stabilizers, not alternative physical assumptions, and must be included in reproducibility records.
