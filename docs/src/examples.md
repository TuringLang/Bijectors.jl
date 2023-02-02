```@setup advi
using Bijectors
```

## Univariate ADVI example
But the real utility of `TransformedDistribution` becomes more apparent when using `transformed(dist, b)` for any bijector `b`. To get the transformed distribution corresponding to the `Beta(2, 2)`, we called `transformed(dist)` before. This is simply an alias for `transformed(dist, bijector(dist))`. Remember `bijector(dist)` returns the constrained-to-constrained bijector for that particular `Distribution`. But we can of course construct a `TransformedDistribution` using different bijectors with the same `dist`. This is particularly useful in something called _Automatic Differentiation Variational Inference (ADVI)_.[2] An important part of ADVI is to approximate a constrained distribution, e.g. `Beta`, as follows:
1. Sample `x` from a `Normal` with parameters `μ` and `σ`, i.e. `x ~ Normal(μ, σ)`.
2. Transform `x` to `y` s.t. `y ∈ support(Beta)`, with the transform being a differentiable bijection with a differentiable inverse (a "bijector")

This then defines a probability density with same _support_ as `Beta`! Of course, it's unlikely that it will be the same density, but it's an _approximation_. Creating such a distribution becomes trivial with `Bijector` and `TransformedDistribution`:

```@repl advi
using StableRNGs: StableRNG
rng = StableRNG(42);
dist = Beta(2, 2)
b = bijector(dist)              # (0, 1) → ℝ
b⁻¹ = inverse(b)                # ℝ → (0, 1)
td = transformed(Normal(), b⁻¹) # x ∼ 𝓝(0, 1) then b(x) ∈ (0, 1)
 x = rand(rng, td)                   # ∈ (0, 1)
```

It's worth noting that `support(Beta)` is the _closed_ interval `[0, 1]`, while the constrained-to-unconstrained bijection, `Logit` in this case, is only well-defined as a map `(0, 1) → ℝ` for the _open_ interval `(0, 1)`. This is of course not an implementation detail. `ℝ` is itself open, thus no continuous bijection exists from a _closed_ interval to `ℝ`. But since the boundaries of a closed interval has what's known as measure zero, this doesn't end up affecting the resulting density with support on the entire real line. In practice, this means that

```@repl advi
td = transformed(Beta())
inverse(td.transform)(rand(rng, td))
```

will never result in `0` or `1` though any sample arbitrarily close to either `0` or `1` is possible. _Disclaimer: numerical accuracy is limited, so you might still see `0` and `1` if you're lucky._

## Multivariate ADVI example
We can also do _multivariate_ ADVI using the `Stacked` bijector. `Stacked` gives us a way to combine univariate and/or multivariate bijectors into a singe multivariate bijector. Say you have a vector `x` of length 2 and you want to transform the first entry using `Exp` and the second entry using `Log`. `Stacked` gives you an easy and efficient way of representing such a bijector.

```@repl advi
using Bijectors: SimplexBijector

# Original distributions
dists = (
    Beta(),
    InverseGamma(),
    Dirichlet(2, 3)
);

# Construct the corresponding ranges
ranges = [];
idx = 1;

for i = 1:length(dists)
    d = dists[i]
    push!(ranges, idx:idx + length(d) - 1)

    global idx
    idx += length(d)
end;

ranges

# Base distribution; mean-field normal
num_params = ranges[end][end]

d = MvNormal(zeros(num_params), ones(num_params));

# Construct the transform
bs = bijector.(dists);     # constrained-to-unconstrained bijectors for dists
ibs = inverse.(bs);            # invert, so we get unconstrained-to-constrained
sb = Stacked(ibs, ranges) # => Stacked <: Bijector

# Mean-field normal with unconstrained-to-constrained stacked bijector
td = transformed(d, sb);
y = rand(td)
0.0 ≤ y[1] ≤ 1.0
0.0 < y[2]
sum(y[3:4]) ≈ 1.0
```

## Normalizing flows
A very interesting application is that of _normalizing flows_.[1] Usually this is done by sampling from a multivariate normal distribution, and then transforming this to a target distribution using invertible neural networks. Currently there are two such transforms available in Bijectors.jl: `PlanarLayer` and `RadialLayer`. Let's create a flow with a single `PlanarLayer`:

```@setup normalizing-flows
using Bijectors
using StableRNGs: StableRNG
rng = StableRNG(42);
```

```@repl normalizing-flows
d = MvNormal(zeros(2), ones(2));
b = PlanarLayer(2)
flow = transformed(d, b)
flow isa MultivariateDistribution
```

That's it. Now we can sample from it using `rand` and compute the `logpdf`, like any other `Distribution`.

```@repl normalizing-flows
y = rand(rng, flow)
logpdf(flow, y)         # uses inverse of `b`
```

Similarily to the multivariate ADVI example, we could use `Stacked` to get a _bounded_ flow:

```@repl normalizing-flows
d = MvNormal(zeros(2), ones(2));
ibs = inverse.(bijector.((InverseGamma(2, 3), Beta())));
sb = stack(ibs...) # == Stacked(ibs) == Stacked(ibs, [i:i for i = 1:length(ibs)]
b = sb ∘ PlanarLayer(2)
td = transformed(d, b);
y = rand(rng, td)
0 < y[1]
0 ≤ y[2] ≤ 1
```

Want to fit the flow?

```@repl normalizing-flows
using Zygote

# Construct the flow.
b = PlanarLayer(2)

# Convenient for extracting parameters and reconstructing the flow.
using Functors
θs, reconstruct = Functors.functor(b);

# Make the objective a `struct` to avoid capturing global variables.
struct NLLObjective{R,D,T}
    reconstruct::R
    basedist::D
    data::T
end

function (obj::NLLObjective)(θs...)
    transformed_dist = transformed(obj.basedist, obj.reconstruct(θs))
    return -sum(Base.Fix1(logpdf, transformed_dist), eachcol(obj.data))
end

# Some random data to estimate the density of.
xs = randn(2, 1000);

# Construct the objective.
f = NLLObjective(reconstruct, MvNormal(2, 1), xs);

# Initial loss.
@info "Initial loss: $(f(θs...))"

# Train using gradient descent.
ε = 1e-3;
for i = 1:100
    ∇s = Zygote.gradient(f, θs...)
    θs = map(θs, ∇s) do θ, ∇
        θ - ε .* ∇
    end
end

# Final loss
@info "Finall loss: $(f(θs...))"

# Very simple check to see if we learned something useful.
samples = rand(transformed(f.basedist, f.reconstruct(θs)), 1000);
mean(eachcol(samples)) # ≈ [0, 0]
cov(samples; dims=2)   # ≈ I
```

We can easily create more complex flows by simply doing `PlanarLayer(10) ∘ PlanarLayer(10) ∘ RadialLayer(10)` and so on.
