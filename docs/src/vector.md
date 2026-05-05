# [The VectorBijectors module](@id vector)

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

## Why does this module exist?

This module is intended primarily for use with probabilistic programming, e.g. DynamicPPL.jl, where vectorised samples are required to satisfy the LogDensityProblems.jl interface.

The core Bijectors.jl interface does indeed contain very similar functionality, but it does not guarantee that `Bijectors.bijector(d)(x)` will always return a vector (in general it can be a scalar or an array of any dimension).
Thus, there is often extra overhead introduced when converting to and from vectorised forms.
It also makes it difficult to correctly handle edge cases, especially when dealing with recursive calls to `bijector` for nested distributions (such as `product_distribution`).
See e.g. https://github.com/TuringLang/DynamicPPL.jl/issues/1142.

## Implementing your own vector bijector

The full VectorBijectors interface consists of the following functions:

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

In practice, if your distribution is a univariate distribution, you will probably only need to implement `scalar_to_scalar_bijector` (see below).

For multivariate and matrix distributions, there are default implementations of the non-linked versions (i.e., `from_vec`, `to_vec`, `vec_length`, and `optic_vec`) which should already be optimal.
However you will have to define the linked versions.
The process of implementing `from_linked_vec` and `to_linked_vec` for a distribution is _very_ similar to the process of implementing `Bijectors.bijector`, so you can consult [the existing documentation for a guide on this](@ref bijectors-defining-examples).

If you have a very customised distribution, you will likely have to implement all the functions yourself.

## Known constant bijectors

Additionally, if your distribution is likely to be used in part of a product distribution, it can lead to substantial performance improvements to overload `has_constant_vec_bijector` to return `true` (but make sure to only do this if the bijector is genuinely constant!):

```@docs
Bijectors.VectorBijectors.has_constant_vec_bijector
```

## Univariate distributions

For univariate distributions the default definition is to generate a bijector that inspects the minimum and maximum of the distribution.
While this will work correctly, it might not be the most performant.
You can manually define the VectorBijectors API for univariate distributions, but it is probably faster to just overload the single function `scalar_to_scalar_bijector`: everything else will be automatically dervied.

```@docs
Bijectors.VectorBijectors.scalar_to_scalar_bijector
```

Univariate distributions tend to fall into one of the following categories:

```@docs
Bijectors.VectorBijectors.TypedIdentity
Bijectors.VectorBijectors.Log
Bijectors.VectorBijectors.Untruncate
```

## Testing

Because the scope of a vector bijector is very well-defined, there is a well-established testing framework to verify correctness of an implementation (`Bijectors.VectorBijectors.test_all()`), which you can use in the test suite.
This function contains additional keyword arguments to control the exact testing procedure.
For example, you can test that the transformations do not cause extra allocations, should you know this to be the case for your bijector (note that this is not always possible).

For more information about generally testing bijectors (and in particular how to test Jacobians for transformations that modify the number of dimensions), see [the documentation on examples of defining bijectors](@ref bijectors-defining-examples).

One of the most tricky parts of testing Bijectors is ensuring that the transforms are compatible with automatic differentiation.
This is important for DynamicPPL: we need to be able to compute the gradient of the log-density with respect to (possibly transformed) parameters, which may include the log-abs-det-Jacobian of the transformation.
The default AD backends tested are ForwardDiff, ReverseDiff, Mooncake, and Enzyme.
It is acceptable to skip tests for a particular backend if there are genuine upstream bugs, especially with ReverseDiff, which is not actively maintained.
However where possible it is best to ensure that all backends are supported, and to use `@test_broken` to mark any known issues with specific backends.
