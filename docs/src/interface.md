# Interface

This page describes the user-facing interface of Bijectors.jl.
You should be able to use all the functions documented here with any bijector defined in Bijectors.jl.

## Transformation

```@docs
transform
transform!
```

Bijectors are also callable objects, so `b(x)` is equivalent to `transform(b, x)`.

## Inverses

```@docs
inverse
```

## Log-absolute determinant of the Jacobian

```@docs
logabsdetjac
logabsdetjac!
logabsdetjacinv
with_logabsdet_jacobian
with_logabsdet_jacobian!
```

## Transform wrappers

### Elementwise transformation

Some transformations are well-defined for different types of inputs, e.g. `exp` can also act elementwise on an `N`-dimensional `Array{<:Real,N}`.
To specify that a transformation should act elementwise, we can wrap it in the `elementwise` wrapper:

```@docs
Bijectors.elementwise
```

### Columnwise transformation

Likewise:

```@docs
Bijectors.columnwise
```

## Working with distributions

```@docs
Bijectors.bijector
Bijectors.link
Bijectors.invlink
Bijectors.logpdf_with_trans
Bijectors.output_size
Bijectors.transformed(d::Distribution, b::Bijector)
Bijectors.ordered
```

## Utilities

```@docs
Bijectors.isinvertible
Bijectors.isclosedform(t::Bijectors.Transform)
```
