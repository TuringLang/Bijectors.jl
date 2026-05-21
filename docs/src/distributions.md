# Usage with distributions

Bijectors provides many utilities for working with probability distributions.

```@example distributions
using Bijectors

dist = LogNormal()
x = rand(dist)
b = bijector(dist)  # bijection (0, ∞) → ℝ

y = b(x)
```

Here, `bijector(d::Distribution)` returns the corresponding constrained-to-unconstrained bijection for `Beta`, which is a log function.
The resulting bijector can be called, just like any other function, to transform samples from the distribution to the unconstrained space.

The function [`link`](@ref) provides a short way of doing the above:

```@example distributions
link(dist, x) ≈ b(x)
```

See [the Turing.jl docs](https://turinglang.org/docs/developers/transforms/distributions/) for more information about how this is used in probabilistic programming.

## Transforming distributions

We can also couple a distribution together with its bijector to create a _transformed_ `Distribution`, i.e. a `Distribution` defined by sampling from a given `Distribution` and then transforming using a given transformation:

```@example distributions
dist = LogNormal()          # support on (0, ∞)
tdist = transformed(dist)   # support on ℝ
```

We can then sample from, and compute the `logpdf` for, the resulting distribution:

```@example distributions
y = rand(tdist)
```

```@example distributions
logpdf(tdist, y)
```

We should expect here that

```julia
logpdf(tdist, y) ≈ logpdf(dist, x) - logabsdetjac(b, x)
```

where `b = bijector(dist)` and `y = b(x)`.

To verify this, we can calculate the value of `x` using the inverse bijector:

```@example distributions
b = bijector(dist)
binv = inverse(b)

x = binv(y)
```

(Because `b` is just a log function, `binv` is an exponential function, i.e. `x = exp(y)`.)

Then we can check the equality:

```@example distributions
logpdf(tdist, y) ≈ logpdf(dist, x) - logabsdetjac(b, x)
```

You can also use [`Bijectors.logpdf_with_trans`](@ref) with the original distribution:

```@example distributions
logpdf_with_trans(dist, x, false) ≈ logpdf(dist, x)
```

```@example distributions
logpdf_with_trans(dist, x, true) ≈ logpdf(tdist, y)
```
