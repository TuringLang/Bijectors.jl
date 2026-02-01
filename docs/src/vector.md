# The VectorBijectors module

The `Bijectors.VectorBijectors` module contains functionality that is very similar to that in the core Bijectors modules, but is specifically focused on converting random samples from distributions to and from **vectors**.

It assumes that there are three forms of samples from a distribution `d` that we are interested in:

 1. **The original form**, which is what `rand(d)` returns.

 2. **A vectorised form**, which is a vector that contains a flattened version of the original form.
 3. **A linked vectorised form**, which is a vector in which:
    
      + each element is independent; and
      + each element is unconstrained (can take any value in ℝ).

Note that because of the independence requirement, the linked vectorised form may have a different dimension to the vectorised form.
For example, when sampling from a `Dirichlet` distribution, the original form is a vector that always sums to 1.
The linked vectorised form will have one element less than the original form, because this constraint is eliminated.

The `Bijectors.VectorBijectors` module provides functionality to convert between these three forms, via the following functions.
Assuming that `x = rand(d)` for some distribution `d`:

  - `to_vec(d)` is a function which converts `x` to the vectorised form
  - `from_vec(d)` is the inverse of `to_vec(d)`
  - `vec_length(d)` returns the length of `to_vec(d)(x)`
  - `optic_vec(d)` returns a vector of optics that describes how each element of `to_vec(d)(x)` is accessed from `x`
  - `to_linked_vec(d)` is a function which converts `x` to the linked vectorised form
  - `from_linked_vec(d)` is the inverse of `to_linked_vec(d)`
  - `linked_vec_length(d)` returns the length of `to_linked_vec(d)(x)`
  - `linked_optic_vec(d)` returns a vector of optics that describes how each element of `to_linked_vec(d)(x)` is accessed from `x` (if possible)

For example:

```julia
julia> using Bijectors.VectorBijectors, Distributions

julia> d = Beta(2, 2);
       x = rand(d);  # x is between 0 and 1
0.5602086057097567

julia> to_vec(d)(x)
1-element Vector{Float64}:
 0.5602086057097567

julia> to_linked_vec(d)(x)
1-element Vector{Float64}:
 0.24200871395677753
```

The bijectors here will also implement `ChangesOfVariables.with_logabsdet_jacobian` as well as `InverseFunctions.inverse`.
See the main Bijectors documentation for more details of this interface.

## When would you use this?

This module is intended primarily for use with probabilistic programming, e.g. DynamicPPL.jl, where vectorised samples are required to satisfy the LogDensityProblems.jl interface.

The core Bijectors.jl interface does indeed contain very similar functionality, but it does not guarantee that `Bijectors.bijector(d)(x)` will always return a vector (in general it can be a scalar or an array of any dimension).
Thus, there is often extra overhead introduced when converting to and from vectorised forms.
It also makes it difficult to correctly handle edge cases, especially when dealing with recursive calls to `bijector` for nested distributions (such as `product_distribution`).
See e.g. https://github.com/TuringLang/DynamicPPL.jl/issues/1142.

## Docstrings

```@docs
Bijectors.VectorBijectors.from_vec
Bijectors.VectorBijectors.to_vec
Bijectors.VectorBijectors.from_linked_vec
Bijectors.VectorBijectors.to_linked_vec
Bijectors.VectorBijectors.vec_length
Bijectors.VectorBijectors.linked_vec_length
Bijectors.VectorBijectors.optic_vec
Bijectors.VectorBijectors.linked_optic_vec
```
