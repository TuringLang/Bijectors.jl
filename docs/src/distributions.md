# Basic usage
Other than the `logpdf_with_trans` methods, the package also provides a more composable interface through the `Bijector` types. Consider for example the one from above with `Beta(2, 2)`.

```julia
julia> using Random; Random.seed!(42);

julia> using Bijectors; using Bijectors: Logit

julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> x = rand(dist)
0.36888689965963756

julia> b = bijector(dist) # bijection (0, 1) → ℝ
Logit{Float64}(0.0, 1.0)

julia> y = b(x)
-0.5369949942509267
```

In this case we see that `bijector(d::Distribution)` returns the corresponding constrained-to-unconstrained bijection for `Beta`, which indeed is a `Logit` with `a = 0.0` and `b = 1.0`. The resulting `Logit <: Bijector` has a method `(b::Logit)(x)` defined, allowing us to call it just like any other function. Comparing with the above example, `b(x) ≈ link(dist, x)`. Just to convince ourselves:

```julia
julia> b(x) ≈ link(dist, x)
true
```

## Transforming distributions

```@setup transformed-dist-simple
using Bijectors
```

We can create a _transformed_ `Distribution`, i.e. a `Distribution` defined by sampling from a given `Distribution` and then transforming using a given transformation:

```@repl transformed-dist-simple
dist = Beta(2, 2)      # support on (0, 1)
tdist = transformed(dist) # support on ℝ

tdist isa UnivariateDistribution
```

We can the then compute the `logpdf` for the resulting distribution:

```@repl transformed-dist-simple
# Some example values
x = rand(dist)
y = tdist.transform(x)

logpdf(tdist, y)
```

When computing `logpdf(tdist, y)` where `tdist` is the _transformed_ distribution corresponding to `Beta(2, 2)`, it makes more semantic sense to compute the pdf of the _transformed_ variable `y` rather than using the "un-transformed" variable `x` to do so, as we do in `logpdf_with_trans`. With that being said, we can also do

```julia
logpdf_forward(tdist, x)
```

We can of course also sample from `tdist`:

```julia
julia> y = rand(td)              # ∈ ℝ
0.999166054552483

julia> x = inverse(td.transform)(y)  # transform back to interval [0, 1]
0.7308945834125756
```


