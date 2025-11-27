# Example: Variational inference

The real utility of `TransformedDistribution` becomes more apparent when using `transformed(dist, b)` for any bijector `b`.
To get the transformed distribution corresponding to the `Beta(2, 2)`, we called `transformed(dist)` before.
This is an alias for `transformed(dist, bijector(dist))`.
Remember `bijector(dist)` returns the constrained-to-constrained bijector for that particular `Distribution`.
But we can of course construct a `TransformedDistribution` using different bijectors with the same `dist`.

This is particularly useful in _Automatic Differentiation Variational Inference (ADVI)_.

## Univariate ADVI

An important part of ADVI is to approximate a constrained distribution, e.g. `Beta`, as follows:

 1. Sample `x` from a `Normal` with parameters `μ` and `σ`, i.e. `x ~ Normal(μ, σ)`.
 2. Transform `x` to `y` s.t. `y ∈ support(Beta)`, with the transform being a differentiable bijection with a differentiable inverse (a "bijector").

This then defines a probability density with the same _support_ as `Beta`!
Of course, it's unlikely that it will be the same density, but it's an _approximation_.

Creating such a distribution can be done with `Bijector` and `TransformedDistribution`:

```@example advi
using Bijectors
using StableRNGs: StableRNG
rng = StableRNG(42)

dist = Beta(2, 2)
b = bijector(dist)                # (0, 1) → ℝ
b⁻¹ = inverse(b)                  # ℝ → (0, 1)
td = transformed(Normal(), b⁻¹)   # x ∼ 𝓝(0, 1) then b(x) ∈ (0, 1)
x = rand(rng, td)                 # ∈ (0, 1)
```

It's worth noting that `support(Beta)` is the _closed_ interval `[0, 1]`, while the constrained-to-unconstrained bijection, `Logit` in this case, is only well-defined as a map `(0, 1) → ℝ` for the _open_ interval `(0, 1)`.
This is of course not an implementation detail.
`ℝ` is itself open, thus no continuous bijection exists from a _closed_ interval to `ℝ`.
But since the boundaries of a closed interval has what's known as measure zero, this doesn't end up affecting the resulting density with support on the entire real line.
In practice, this means that

```@example advi
td = transformed(Beta())
inverse(td.transform)(rand(rng, td))
```

will never result in `0` or `1` though any sample arbitrarily close to either `0` or `1` is possible.
_Disclaimer: numerical accuracy is limited, so you might still see `0` and `1` if you're 'lucky'._

## Multivariate ADVI example

We can also do _multivariate_ ADVI using the `Stacked` bijector.
`Stacked` gives us a way to combine univariate and/or multivariate bijectors into a singe multivariate bijector.
Say you have a vector `x` of length 2 and you want to transform the first entry using `Exp` and the second entry using `Log`.
`Stacked` gives you an easy and efficient way of representing such a bijector.

```@example advi
using Bijectors: SimplexBijector

# Original distributions
dists = (Beta(), InverseGamma(), Dirichlet(2, 3))

# Construct the corresponding ranges
function make_ranges(dists)
    ranges = []
    idx = 1
    for i in 1:length(dists)
        d = dists[i]
        push!(ranges, idx:(idx + length(d) - 1))
        idx += length(d)
    end
    return ranges
end

ranges = make_ranges(dists)
ranges
```

```@example advi
# Base distribution; mean-field normal
num_params = ranges[end][end]

d = MvNormal(zeros(num_params), ones(num_params));

# Construct the transform
bs = bijector.(dists)       # constrained-to-unconstrained bijectors for dists
ibs = inverse.(bs)          # invert, so we get unconstrained-to-constrained
sb = Stacked(ibs, ranges)   # => Stacked <: Bijector

# Mean-field normal with unconstrained-to-constrained stacked bijector
td = transformed(d, sb)
y = rand(td)
```

As can be seen from this, we now have a `y` for which `0.0 ≤ y[1] ≤ 1.0`, `0.0 < y[2]`, and `sum(y[3:4]) ≈ 1.0`.
