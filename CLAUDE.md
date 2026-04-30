# Bijectors.jl

Bijective transformations for probability distributions, used primarily by Turing.jl / DynamicPPL.jl.

## Active codebase

The actively developed API is the `VectorBijectors` submodule in `src/vector/`. This is what DynamicPPL uses for converting distribution samples to and from flat unconstrained real vectors (needed for HMC/NUTS).

The code in `src/` outside of `src/vector/` is largely legacy. It still works and is used internally by `VectorBijectors`, but new development should focus on the `VectorBijectors` API.

## Key documentation

  - `docs/src/vector.md` — the VectorBijectors developer guide: API overview, how to add support for new distributions, the `ScalarToScalarBijector` interface, `has_constant_vec_bijector`, and testing
  - `docs/src/defining.md` and `docs/src/defining_examples.md` — guide to defining new bijectors (applies to both legacy and VectorBijectors implementations)

## Testing

Run tests with `julia --project -e 'using Pkg; Pkg.test()'`.

Use `Bijectors.VectorBijectors.test_all(d)` to test a new distribution's VectorBijectors implementation. It checks roundtrips, type stability, vector lengths, optics, allocations, log-Jacobian correctness (against ForwardDiff), and AD compatibility across backends.

AD backends tested: ForwardDiff (reference), ReverseDiff, Mooncake, Enzyme. It is acceptable to skip a backend for genuine upstream bugs (especially ReverseDiff), but prefer `@test_broken` over silently omitting tests.

## Code review checklist

When reviewing changes to VectorBijectors:

  - New distributions need the linked transforms (`to_linked_vec`, `from_linked_vec`, `linked_vec_length`, `linked_optic_vec`). The unlinked versions (`to_vec`, `from_vec`, `vec_length`, `optic_vec`) have defaults that are usually sufficient.
  - Univariate distributions usually only need `scalar_to_scalar_bijector(d)`.
  - If a distribution will appear in `product_distribution`, check whether `has_constant_vec_bijector` should return `true` (it can if the bijector doesn't depend on parameter values).
  - `linked_optic_vec` should return `nothing` for elements without a one-to-one correspondence to the original sample (e.g. simplex, posdef).
  - `with_logabsdet_jacobian(to_vec(d), x)` must always return zero log-Jacobian.
  - Verify `test_all(d)` passes, including AD tests.
