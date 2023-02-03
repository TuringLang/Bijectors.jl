# Bijectors.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/Bijectors.jl/stable)
[![Interface tests](https://github.com/TuringLang/Bijectors.jl/workflows/Interface%20tests/badge.svg?branch=master)](https://github.com/TuringLang/Bijectors.jl/actions?query=workflow%3A%22Interface+tests%22+branch%3Amaster)
[![AD tests](https://github.com/TuringLang/Bijectors.jl/workflows/AD%20tests/badge.svg?branch=master)](https://github.com/TuringLang/Bijectors.jl/actions?query=workflow%3A%22AD+tests%22+branch%3Amaster)

This package implements a set of functions for transforming constrained random variables (e.g. simplexes, intervals) to Euclidean space. The 3 main functions implemented in this package are the `link`, `invlink` and `logpdf_with_trans` for a number of distributions. The distributions supported are:
1. `RealDistribution`: `Union{Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal, NormalCanon, TDist}`,
2. `PositiveDistribution`: `Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull}`,
3. `UnitDistribution`: `Union{Beta, KSOneSided, NoncentralBeta}`,
4. `SimplexDistribution`: `Union{Dirichlet}`,
5. `PDMatDistribution`: `Union{InverseWishart, Wishart}`, and
6. `TransformDistribution`: `Union{T, Truncated{T}} where T<:ContinuousUnivariateDistribution`.

All exported names from the [Distributions.jl](https://github.com/TuringLang/Bijectors.jl) package are reexported from `Bijectors`.

Bijectors.jl also provides a nice interface for working with these maps: composition, inversion, etc.
The following table lists mathematical operations for a bijector and the corresponding code in Bijectors.jl.

| Operation                          | Method          | Automatic |
|:------------------------------------:|:-----------------:|:-----------:|
| `b ↦ b⁻¹`                                      | `inverse(b)`                | ✓         |
| `(b₁, b₂) ↦ (b₁ ∘ b₂)`                         | `b₁ ∘ b₂`               | ✓         |
| `(b₁, b₂) ↦ [b₁, b₂]`                          | `stack(b₁, b₂)`         | ✓         |
| `x ↦ b(x)`                                     | `b(x)`                  | ×         |
| `y ↦ b⁻¹(y)`                                   | `inverse(b)(y)`             | ×         |
| `x ↦ log｜det J(b, x)｜`                       | `logabsdetjac(b, x)`    | AD        |
| `x ↦ b(x), log｜det J(b, x)｜`                 | `with_logabsdet_jacobian(b, x)`         | ✓         |
| `p ↦ q := b_* p`                                | `q = transformed(p, b)` | ✓         |
| `y ∼ q`                                        | `y = rand(q)`           | ✓         |
| `p ↦ b` such that `support(b_* p) = ℝᵈ`               | `bijector(p)`           | ✓         |
| `(x ∼ p, b(x), log｜det J(b, x)｜, log q(y))` | `forward(q)`            | ✓         |

In this table, `b` denotes a `Bijector`, `J(b, x)` denotes the Jacobian of `b` evaluated at `x`, `b_*` denotes the [push-forward](https://www.wikiwand.com/en/Pushforward_measure) of `p` by `b`, and `x ∼ p` denotes `x` sampled from the distribution with density `p`.

The "Automatic" column in the table refers to whether or not you are required to implement the feature for a custom `Bijector`. "AD" refers to the fact that this can be implemented "automatically" using automatic differentiation, e.g. ForwardDiff.jl.

## Functions

1. `link`: maps a sample of a random distribution `dist` from its support to a value in ℝⁿ. Example:

```julia
julia> using Bijectors

julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> x = rand(dist)
0.7472542331020509

julia> y = link(dist, x)
1.084021356473311
```

2. `invlink`: the inverse of the `link` function. Example:

```julia
julia> z = invlink(dist, y)
0.7472542331020509

julia> x ≈ z
true
```

3. `logpdf_with_trans`: finds `log` of the (transformed) probability density function of a distribution `dist` at a sample `x`. Example:

```julia
julia> using Bijectors

julia> dist = Dirichlet(2, 3)
Dirichlet{Float64}(alpha=[3.0, 3.0])

julia> x = rand(dist)
2-element Array{Float64,1}:
 0.46094823621110165
 0.5390517637888984

julia> logpdf_with_trans(dist, x, false) # ignoring the transformation
0.6163709733893024

julia> logpdf_with_trans(dist, x, true) # considering the transformation
-0.7760422307471244
```

## `Bijector` interface
A `Bijector` is a differentiable bijection with a differentiable inverse. That's basically it.

The primary application of `Bijector`s is the (very profitable) business of transforming (usually continuous) probability densities. If we transfrom a random variable `x ~ p(x)` to `y = b(x)` where `b` is a `Bijector`, we also get a canonical density `q(y) = p(b⁻¹(y)) |det J(b⁻¹, y)|` for `y`. Here `J(b⁻¹, y)` is the Jacobian of the inverse transform evaluated at `y`. `q` is also known as the _push-forward_ of `p` by `b` in measure theory.

There's plenty of different reasons why one would want to do something like this. It can be because your `p` has non-zero probability (support) on a closed interval `[a, b]` and you want to use AD without having to worry about reaching the boundary. E.g. `Beta` has support `[0, 1]` so if we could transform `p = Beta` into a density `q` with support on ℝ, we could instead compute the derivative of `logpdf(q, y)` wrt. `y`, and then transform back `x = b⁻¹(y)`. This is very useful for certain inference methods, e.g. Hamiltonian Monte-Carlo, where we need to take the derivative of the logpdf-computation wrt. input.

Another use-case is constructing a _parameterized_ `Bijector` and consider transforming a "simple" density, e.g. `MvNormal`, to match a more complex density. One class of such bijectors is _Normalizing Flows (NFs)_ which are compositions of differentiable and invertible neural networks, i.e. composition of a particular family of parameterized bijectors.[1] We'll see an example of this later on.

### Basic usage
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

#### Inversion

What about `invlink`?

```julia
julia> b⁻¹ = inverse(b)
Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0))

julia> b⁻¹(y)
0.3688868996596376

julia> b⁻¹(y) ≈ invlink(dist, y)
true
```

Pretty neat, huh? `Inverse{Logit}` is also a `Bijector` where we've defined `(ib::Inverse{<:Logit})(y)` as the inverse transformation of `(b::Logit)(x)`. Note that it's not always the case that `inverse(b) isa Inverse`, e.g. the inverse of `Exp` is simply `Log` so `inverse(Exp()) isa Log` is true.

#### Composition
Also, we can _compose_ bijectors:

```julia
julia> id_y = (b ∘ b⁻¹)
Composed{Tuple{Inverse{Logit{Float64},0},Logit{Float64}},0}((Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0)), Logit{Float64}(0.0, 1.0)))

julia> id_y(y) ≈ y
true
```
    
And since `Composed isa Bijector`:

```julia
julia> id_x = inverse(id_y)
Composed{Tuple{Inverse{Logit{Float64},0},Logit{Float64}},0}((Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0)), Logit{Float64}(0.0, 1.0)))

julia> id_x(x) ≈ x
true
```

#### `logpdf` of `TransformedDistribution`
This far we've seen that we can replicate the functionality provided by `link` and `invlink`. To replicate `logpdf_with_trans` we instead provide a `TransformedDistribution <: Distribution` implementing the `Distribution` interface from Distributions.jl:

```julia
julia> using Bijectors: TransformedDistribution

julia> td = transformed(dist)
TransformedDistribution{Beta{Float64},Logit{Float64},Univariate}(
dist: Beta{Float64}(α=2.0, β=2.0)
transform: Logit{Float64}(0.0, 1.0)
)


julia> td isa UnivariateDistribution
true

julia> logpdf(td, y)
-1.123311289915276

julia> logpdf_with_trans(dist, x, true)
-1.123311289915276
```

When computing `logpdf(td, y)` where `td` is the _transformed_ distribution corresponding to `Beta(2, 2)`, it makes more semantic sense to compute the pdf of the _transformed_ variable `y` rather than using the "un-transformed" variable `x` to do so, as we do in `logpdf_with_trans`. With that being said, we can also do

```julia
julia> logpdf_forward(td, x)
-1.123311289915276
```

#### `logabsdetjac` and `with_logabsdet_jacobian`

In the computation of both `logpdf` and `logpdf_forward` we need to compute `log(abs(det(jacobian(inverse(b), y))))` and `log(abs(det(jacobian(b, x))))`, respectively. This computation is available using the `logabsdetjac` method

```julia
julia> logabsdetjac(b⁻¹, y)
-1.4575353795716655

julia> logabsdetjac(b, x)
1.4575353795716655
```

Notice that

```julia
julia> logabsdetjac(b, x) ≈ -logabsdetjac(b⁻¹, y)
true
```

which is always the case for a differentiable bijection with differentiable inverse. Therefore if you want to compute `logabsdetjac(b⁻¹, y)` and we know that `logabsdetjac(b, b⁻¹(y))` is actually more efficient, we'll return `-logabsdetjac(b, b⁻¹(y))` instead. For some bijectors it might be easy to compute, say, the forward pass `b(x)`, but expensive to compute `b⁻¹(y)`. Because of this you might want to avoid doing anything "backwards", i.e. using `b⁻¹`. This is where `with_logabsdet_jacobian` comes to good use:

```julia
julia> with_logabsdet_jacobian(b, x)
(-0.5369949942509267, 1.4575353795716655)
```

Similarily

```julia
julia> with_logabsdet_jacobian(inverse(b), y)
(0.3688868996596376, -1.4575353795716655)
```

In fact, the purpose of `with_logabsdet_jacobian` is to just _do the right thing_, not necessarily "forward". In this function we'll have access to both the original value `x` and the transformed value `y`, so we can compute `logabsdetjac(b, x)` in either direction. Furthermore, in a lot of cases we can re-use a lot of the computation from `b(x)` in the computation of `logabsdetjac(b, x)`, or vice-versa. `with_logabsdet_jacobian(b, x)` will take advantage of such opportunities (if implemented).

#### Sampling from `TransformedDistribution`
At this point we've only shown that we can replicate the existing functionality. But we said `TransformedDistribution isa Distribution`, so we also have `rand`:

```julia
julia> y = rand(td)              # ∈ ℝ
0.999166054552483

julia> x = inverse(td.transform)(y)  # transform back to interval [0, 1]
0.7308945834125756
```

This can be quite convenient if you have computations assuming input to be on the real line.

#### Univariate ADVI example
But the real utility of `TransformedDistribution` becomes more apparent when using `transformed(dist, b)` for any bijector `b`. To get the transformed distribution corresponding to the `Beta(2, 2)`, we called `transformed(dist)` before. This is simply an alias for `transformed(dist, bijector(dist))`. Remember `bijector(dist)` returns the constrained-to-constrained bijector for that particular `Distribution`. But we can of course construct a `TransformedDistribution` using different bijectors with the same `dist`. This is particularly useful in something called _Automatic Differentiation Variational Inference (ADVI)_.[2] An important part of ADVI is to approximate a constrained distribution, e.g. `Beta`, as follows:
1. Sample `x` from a `Normal` with parameters `μ` and `σ`, i.e. `x ~ Normal(μ, σ)`.
2. Transform `x` to `y` s.t. `y ∈ support(Beta)`, with the transform being a differentiable bijection with a differentiable inverse (a "bijector")

This then defines a probability density with same _support_ as `Beta`! Of course, it's unlikely that it will be the same density, but it's an _approximation_. Creating such a distribution becomes trivial with `Bijector` and `TransformedDistribution`:

```julia
julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> b = bijector(dist)              # (0, 1) → ℝ
Logit{Float64}(0.0, 1.0)

julia> b⁻¹ = inverse(b)                    # ℝ → (0, 1)
Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0))

julia> td = transformed(Normal(), b⁻¹) # x ∼ 𝓝(0, 1) then b(x) ∈ (0, 1)
TransformedDistribution{Normal{Float64},Inverse{Logit{Float64},0},Univariate}(
dist: Normal{Float64}(μ=0.0, σ=1.0)
transform: Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0))
)


julia> x = rand(td)                    # ∈ (0, 1)
0.538956748141868
```

It's worth noting that `support(Beta)` is the _closed_ interval `[0, 1]`, while the constrained-to-unconstrained bijection, `Logit` in this case, is only well-defined as a map `(0, 1) → ℝ` for the _open_ interval `(0, 1)`. This is of course not an implementation detail. `ℝ` is itself open, thus no continuous bijection exists from a _closed_ interval to `ℝ`. But since the boundaries of a closed interval has what's known as measure zero, this doesn't end up affecting the resulting density with support on the entire real line. In practice, this means that

```julia
td = transformed(Beta())

inverse(td.transform)(rand(td))
```

will never result in `0` or `1` though any sample arbitrarily close to either `0` or `1` is possible. _Disclaimer: numerical accuracy is limited, so you might still see `0` and `1` if you're lucky._

### Multivariate ADVI example
We can also do _multivariate_ ADVI using the `Stacked` bijector. `Stacked` gives us a way to combine univariate and/or multivariate bijectors into a singe multivariate bijector. Say you have a vector `x` of length 2 and you want to transform the first entry using `Exp` and the second entry using `Log`. `Stacked` gives you an easy and efficient way of representing such a bijector.

```julia
julia> Random.seed!(42);

julia> using Bijectors: Exp, Log, SimplexBijector

julia> # Original distributions
       dists = (
           Beta(),
           InverseGamma(),
           Dirichlet(2, 3)
       );

julia> # Construct the corresponding ranges
       ranges = [];

julia> idx = 1;

julia> for i = 1:length(dists)
           d = dists[i]
           push!(ranges, idx:idx + length(d) - 1)

           global idx
           idx += length(d)
       end;

julia> ranges
3-element Array{Any,1}:
 1:1
 2:2
 3:4

julia> # Base distribution; mean-field normal
       num_params = ranges[end][end]
4

julia> d = MvNormal(zeros(num_params), ones(num_params))
DiagNormal(
dim: 4
μ: [0.0, 0.0, 0.0, 0.0]
Σ: [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
)


julia> # Construct the transform
       bs = bijector.(dists)     # constrained-to-unconstrained bijectors for dists
(Logit{Float64}(0.0, 1.0), Log{0}(), SimplexBijector{true}())

julia> ibs = inverse.(bs)            # invert, so we get unconstrained-to-constrained
(Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0)), Exp{0}(), Inverse{SimplexBijector{true},1}(SimplexBijector{true}()))

julia> sb = Stacked(ibs, ranges) # => Stacked <: Bijector
Stacked{Tuple{Inverse{Logit{Float64},0},Exp{0},Inverse{SimplexBijector{true},1}},3}((Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0)), Exp{0}(), Inverse{SimplexBijector{true},1}(SimplexBijector{true}())), (1:1, 2:2, 3:4))

julia> # Mean-field normal with unconstrained-to-constrained stacked bijector
       td = transformed(d, sb);

julia> y = rand(td)
4-element Array{Float64,1}:
 0.36446726136766217
 0.6412195576273355 
 0.5067884173521743 
 0.4932115826478257 

julia> 0.0 ≤ y[1] ≤ 1.0   # => true
true

julia> 0.0 < y[2]         # => true
true

julia> sum(y[3:4]) ≈ 1.0  # => true
true
```

### Normalizing flows
A very interesting application is that of _normalizing flows_.[1] Usually this is done by sampling from a multivariate normal distribution, and then transforming this to a target distribution using invertible neural networks. Currently there are two such transforms available in Bijectors.jl: `PlanarLayer` and `RadialLayer`. Let's create a flow with a single `PlanarLayer`:

```julia
julia> d = MvNormal(zeros(2), ones(2));

julia> b = PlanarLayer(2)
PlanarLayer{Array{Float64,2},Array{Float64,1}}([1.77786; -1.1449], [-0.468606; 0.156143], [-2.64199])

julia> flow = transformed(d, b)
TransformedDistribution{MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Array{Float64,1}},PlanarLayer{Array{Float64,2},Array{Float64,1}},Multivariate}(
dist: DiagNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0 0.0; 0.0 1.0]
)

transform: PlanarLayer{Array{Float64,2},Array{Float64,1}}([1.77786; -1.1449], [-0.468606; 0.156143], [-2.64199])
)


julia> flow isa MultivariateDistribution
true
```

That's it. Now we can sample from it using `rand` and compute the `logpdf`, like any other `Distribution`.

```julia
julia> y = rand(flow)
2-element Array{Float64,1}:
 1.3337915588180933
 1.010861989639227 

julia> logpdf(flow, y)         # uses inverse of `b`
-2.8996106373788293

julia> x = rand(flow.dist)
2-element Array{Float64,1}:
 0.18702790710363  
 0.5181487878771377

julia> logpdf_forward(flow, x) # more efficent and accurate
-1.9813114667203335
```

Similarily to the multivariate ADVI example, we could use `Stacked` to get a _bounded_ flow:

```julia
julia> d = MvNormal(zeros(2), ones(2));

julia> ibs = inverse.(bijector.((InverseGamma(2, 3), Beta())));

julia> sb = stack(ibs...) # == Stacked(ibs) == Stacked(ibs, [i:i for i = 1:length(ibs)]
Stacked{Tuple{Exp{0},Inverse{Logit{Float64},0}},2}((Exp{0}(), Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0))), (1:1, 2:2))

julia> b = sb ∘ PlanarLayer(2)
Composed{Tuple{PlanarLayer{Array{Float64,2},Array{Float64,1}},Stacked{Tuple{Exp{0},Inverse{Logit{Float64},0}},2}},1}((PlanarLayer{Array{Float64,2},Array{Float64,1}}([1.49138; 0.367563], [-0.886205; 0.684565], [-1.59058]), Stacked{Tuple{Exp{0},Inverse{Logit{Float64},0}},2}((Exp{0}(), Inverse{Logit{Float64},0}(Logit{Float64}(0.0, 1.0))), (1:1, 2:2))))

julia> td = transformed(d, b);

julia> y = rand(td)
2-element Array{Float64,1}:
 2.6493626783431035
 0.1833391433092443

julia> 0 < y[1]
true

julia> 0 ≤ y[2] ≤ 1
true
```

Want to fit the flow?

```julia
julia> using Tracker

julia> b = PlanarLayer(2, param)                  # construct parameters using `param`
PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}}([-1.05099; 0.502079] (tracked), [-0.216248; -0.706424] (tracked), [-4.33747] (tracked))

julia> flow = transformed(d, b)
TransformedDistribution{MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Array{Float64,1}},PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}},Multivariate}(
dist: DiagNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0 0.0; 0.0 1.0]
)

transform: PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}}([-1.05099; 0.502079] (tracked), [-0.216248; -0.706424] (tracked), [-4.33747] (tracked))
)


julia> rand(flow)
Tracked 2-element Array{Float64,1}:
  0.5992818950827451
 -0.6264187818605164

julia> x = rand(flow.dist)
2-element Array{Float64,1}:
 -0.37240087577993225
  0.36901028455183293

julia> Tracker.back!(logpdf_forward(flow, x), 1.0) # backprob

julia> Tracker.grad(b.w)
2×1 Array{Float64,2}:
 -0.00037431072968105417
  0.0013039074681623036
```

We can easily create more complex flows by simply doing `PlanarLayer(10) ∘ PlanarLayer(10) ∘ RadialLayer(10)` and so on.

In those cases, it might be useful to use Flux.jl's `Flux.params` to extract the parameters:
```julia
julia> using Flux

julia> Flux.params(flow)
Params([[-1.05099; 0.502079] (tracked), [-0.216248; -0.706424] (tracked), [-4.33747] (tracked)])
```

Another useful function is the `forward(d::Distribution)` method. It is similar to `with_logabsdet_jacobian(b::Bijector, x)` in the sense that it does a forward pass of the entire process "sample then transform" and returns all the most useful quantities in process using the most efficent computation path.

```julia
julia> x, y, logjac, logpdf_y = forward(flow) # sample + transform and returns all the useful quantities in one pass
(x = [-0.839739, 0.169613], y = [-0.810354, 0.963392] (tracked), logabsdetjac = -0.0017416108706436628 (tracked), logpdf = -2.203100286792651 (tracked))
```

This method is for example useful when computing quantities such as the _expected lower bound (ELBO)_ between this transformed distribution and some other joint density. If no analytical expression is available, we have to approximate the ELBO by a Monte Carlo estimate. But one term in the ELBO is the entropy of the base density, which we _do_ know analytically in this case. Using the analytical expression for the entropy and then using a monte carlo estimate for the rest of the terms in the ELBO gives an estimate with lower variance than if we used the monte carlo estimate for the entire expectation.


# Bibliography
1. Rezende, D. J., & Mohamed, S. (2015). Variational Inference With Normalizing Flows. [arXiv:1505.05770](https://arxiv.org/abs/1505.05770v6).
2. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2016). Automatic Differentiation Variational Inference. [arXiv:1603.00788](https://arxiv.org/abs/1603.00788v1).
