# Bijectors.jl

## Usage

A very simple example of a "bijector"/diffeomorphism, i.e. a differentiable transformation with a differentiable inverse, is the `exp` function:
- The inverse of `exp` is `log`.
- The derivative of `exp` at an input `x` is simply `exp(x)`, hence `logabsdetjac` is simply `x`.

```@repl usage
using Bijectors
transform(exp, 1.0)
logabsdetjac(exp, 1.0)
with_logabsdet_jacobian(exp, 1.0)
```

Some transformations is well-defined for different types of inputs, e.g. `exp` can also act elementwise on a `N`-dimensional `Array{<:Real,N}`. To specify that a transformation should be acting elementwise, we use the [`elementwise`](@ref) method:

```@repl usage
x = ones(2, 2)
transform(elementwise(exp), x)
logabsdetjac(elementwise(exp), x)
with_logabsdet_jacobian(elementwise(exp), x)
```

These methods also work nicely for compositions of transformations:

```@repl usage
transform(elementwise(log ∘ exp), x)
```

Unlike `exp`, some transformations have parameters affecting the resulting transformation they represent, e.g. `Logit` has two parameters `a` and `b` representing the lower- and upper-bound, respectively, of its domain:

```@repl usage
using Bijectors: Logit

f = Logit(0.0, 1.0)
f(rand()) # takes us from `(0, 1)` to `(-∞, ∞)`
```

## User-facing methods

Without mutation:

```@docs
transform
logabsdetjac
```

```julia
with_logabsdet_jacobian
```

With mutation:

```@docs
transform!
logabsdetjac!
with_logabsdet_jacobian!
```

## Implementing a transformation

Any callable can be made into a bijector by providing an implementation of `with_logabsdet_jacobian(b, x)`.

You can also optionally implement [`transform`](@ref) and [`logabsdetjac`](@ref) to avoid redundant computations. This is usually only worth it if you expect `transform` or `logabsdetjac` to be used heavily without the other.

Similarly with the mutable versions [`with_logabsdet_jacobian!`](@ref), [`transform!`](@ref), and [`logabsdetjac!`](@ref).

## Working with Distributions.jl

```@docs
Bijectors.bijector
Bijectors.transformed(d::Distribution, b::Bijector)
```

## Utilities

```@docs
Bijectors.elementwise
Bijectors.isinvertible
Bijectors.isclosedform(t::Bijectors.Transform)
Bijectors.invertible
Bijectors.NotInvertible
Bijectors.Invertible
```

## API

```@docs
Bijectors.Transform
Bijectors.Bijector
Bijectors.Inverse
```

## Bijectors

```@docs
Bijectors.CorrBijector
Bijectors.LeakyReLU
Bijectors.Stacked
Bijectors.RationalQuadraticSpline
Bijectors.Coupling
Bijectors.OrderedBijector
Bijectors.NamedTransform
Bijectors.NamedCoupling
```
