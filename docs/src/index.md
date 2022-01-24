# Bijectors.jl

Documentation for Bijectors.jl

## Usage

A very simple example of a "bijector"/diffeomorphism, i.e. a differentiable transformation with a differentiable inverse, is the `exp` function:
- The inverse of `exp` is `log`.
- The derivative of `exp` at an input `x` is simply `exp(x)`, hence `logabsdetjac` is simply `x`.

```@repl usage
using Bijectors
transform(exp, 1.0)
logabsdetjac(exp, 1.0)
forward(exp, 1.0)
```

If you want to instead transform a collection of inputs, you can use the `batch` method from Batching.jl to inform Bijectors.jl that the input now represents a collection of inputs rather than a single input:

```@repl usage
xs = batch(ones(2));
transform(exp, xs)
logabsdetjac(exp, xs)
forward(exp, xs)
```

Some transformations is well-defined for different types of inputs, e.g. `exp` can also act elementwise on a `N`-dimensional `Array{<:Real,N}`. To specify that a transformation should be acting elementwise, we use the [`elementwise`](@ref) method:

```@repl usage
x = ones(2, 2)
transform(elementwise(exp), x)
logabsdetjac(elementwise(exp), x)
forward(elementwise(exp), x)
```

And batched versions:

```@repl usage
xs = batch(ones(2, 2, 3));
transform(elementwise(exp), xs)
logabsdetjac(elementwise(exp), xs)
forward(elementwise(exp), xs)
```

These methods also work nicely for compositions of transformations:

```@repl usage
transform(elementwise(log ∘ exp), xs)
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
forward(b, x)
```

With mutation:

```@docs
transform!
logabsdetjac!
forward!
```

## Implementing a transformation

Any callable can be made into a bijector by providing an implementation of [`forward(b, x)`](@ref), which is done by overloading

```@docs
Bijectors.forward_single
```

where

```@docs
Bijectors.transform_single
Bijectors.logabsdetjac_single
```

You can then optionally implement `transform` and `logabsdetjac` to avoid redundant computations. This is usually only worth it if you expect `transform` or `logabsdetjac` to be used heavily without the other.

Note that a _user_ of the bijector should generally be using [`forward(b, x)`](@ref) rather than calling [`forward_single`](@ref) directly.

To implement "batched" versions of the above functionalities, i.e. methods which act on a _collection_ of inputs rather than a single input, you can overload the following method:

```@docs
Bijectors.forward_multiple
```

And similarly, if you want to specialize [`transform`](@ref) and [`logabsdetjac`](@ref), you can implement

```@docs
Bijectors.transform_multiple
Bijectors.logabsdetjac_multiple
```

### Mutability

There are also _mutable_ versions of all of the above:

```@docs
Bijectors.forward_single!
Bijectors.forward_multiple!
Bijectors.transform_single!
Bijectors.transform_multiple!
Bijectors.logabsdetjac_single!
Bijectors.logabsdetjac_multiple!
```

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
```

## API

```@docs
Bijectors.Transform
Bijectors.Bijector
Bijectors.Inverse
```
