# Bijectors.jl

[![Build Status](https://travis-ci.org/TuringLang/Bijectors.jl.svg?branch=master)](https://travis-ci.org/TuringLang/Bijectors.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/mvfs8eio2cscwk1m?svg=true)](https://ci.appveyor.com/project/TuringLang/bijectors-jl)


This package implements a set of functions for transforming constrained random variables (e.g. simplexes, intervals) to Euclidean space. The 3 main functions implemented in this package are the `link`, `invlink` and `logpdf_with_trans` for a number of distributions. The distributions supported are:
1. `RealDistribution`: `Union{Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal, NormalCanon, TDist}`,
2. `PositiveDistribution`: `Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull}`,
3. `UnitDistribution`: `Union{Beta, KSOneSided, NoncentralBeta}`,
4. `SimplexDistribution`: `Union{Dirichlet}`,
5. `PDMatDistribution`: `Union{InverseWishart, Wishart}`, and
6. `TransformDistribution`: `Union{T, Truncated{T}} where T<:ContinuousUnivariateDistribution`.

All exported names from the [Distributions.jl](https://github.com/TuringLang/Bijectors.jl) package are reexported from `Bijectors`.

## Functions

1. `link`: maps a sample of a random distribution `dist` from its support to a value in R^n. Example:

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
0.6543406780096065

julia> x == z
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
Other than the `logpdf_with_trans` methods, the package also provides a more composable interface through the `Bijector` types. Consider for example the from above with `Beta(2, 2)`.

```julia
julia> using Bijectors; using Bijectors: Logit

julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> x = rand(dist)
0.7173326646959575

julia> b = bijector(dist) # bijection (0, 1) → ℝ
Logit{Float64}(0.0, 1.0)

julia> y = b(x)
0.9312689879144197
```

In this case we see that `bijector(d::Distribution)` returns the corresponding constrained-to-unconstrained bijection for `Beta`, which indeed is a `Logit` with `a = 0.0` and `b = 1.0`. The resulting `Logit <: Bijector` has a method `(b::Logit)(x)` defined, allowing us to call it just like any other function. Comparing with the above example, `b(x) == link(dist, x)`. Just to convince ourselves:

```julia
julia> b(x) == link(dist, x)
true
```

What about `invlink`?

```julia
julia> b⁻¹ = inv(b)
Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0))

julia> b⁻¹(y)
0.7173326646959575

julia> b⁻¹(y) == invlink(dist, y)
true
```

Pretty neat, huh? `Inversed{Logit}` is also a `Bijector` where we've defined `(ib::Inversed{<:Logit})(y)` as the inverse transformation of `(b::Logit)(x)`. Note that it's not always the case that `inv(b) isa Inversed`, e.g. the inverse of `Exp` is simply `Log` so `inv(Exp()) isa Log` is true. Aslo, we can _compose_ bijectors:

```julia
julia> id = (b ∘ b⁻¹)
Composed{Tuple{Inversed{Logit{Float64}},Logit{Float64}}}((Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0)), Logit{Float64}(0.0, 1.0)))

julia> id(y) == y
true
```

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
-1.0577727579778098

julia> logpdf_with_trans(dist, x, true)
-1.05777275797781
```

When computing `logpdf(td, y)` where `td` is the _transformed_ distribution corresponding to `Beta(2, 2)`, it makes more semantic sense to compute the pdf of the _transformed_ variable `y` rather than using the "un-transformed" variable `x` to do so, as we do in `logpdf_with_trans`. With that being said, we can also do

```julia
julia> logpdf_forward(td, x)
-1.05777275797781
```

At this point we've only shown that we can replicate the existing functionality. But we said `TransformedDistribution isa Distribution`, so we also have `rand`:

```julia
julia> y = rand(td)              # ∈ ℝ
-0.5231573469209508

julia> x = inv(td.transform)(y)  # transform back to interval [0, 1]
0.37211423725902915
```

This can be quite convenient if you have computations assuming input to be on the real line.

But the real utility of `TransformedDistribution` becomes more apparent when using `transformed(dist, b)` for any bijector `b`. To get the transformed distribution corresponding to the `Beta(2, 2)`, we called `transformed(dist)` before. This is simply an alias for `transformed(dist, bijector(dist))`. Remember `bijector(dist)` returns the constrained-to-constrained bijector for that particular `Distribution`. But we can of course construct a `TransformedDistribution` using different bijectors for the same distribution! It's particular useful in something called _Automatic Derivative Variational Inference (ADVI)_.(INSERT REF) An important part of this to approximate a constrained distribution, e.g. `Beta`, as follows:
1. Sample `x` from a `Normal` with parameters `μ` and `σ`, i.e. `x ~ Normal(μ, σ)`.
2. Transform `x` to `y` s.t. `y ∈ support(Beta)`, with the transform being a differentiable bijection with a differentiable inverse (a "bijector")
This then defines a probability density with support same as `Beta`! Of course, it's unlikely that it will be the same, but it's an _approximation_. Creating such a distribution becomes trivial with `Bijector` and `TransformedDistribution`:

```julia
julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> b = bijector(dist)              # (0, 1) → ℝ

julia> b⁻¹ = inv(b)                    # ℝ → (0, 1)
Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0))

julia> td = transformed(Normal(), b⁻¹) # x ∼ 𝓝(0, 1) then b(x) ∈ (0, 1)
TransformedDistribution{Normal{Float64},Inversed{Logit{Float64}},Univariate}(
dist: Normal{Float64}(μ=0.0, σ=1.0)
transform: Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0))
)

julia> x = rand(td)                    # ∈ (0, 1)
0.37786466412061664
```

### Normalizing flows
A very interesting application is that of _normalizing flows_.(INSERT REF) Usually this is done by sampling from a multivariate normal distribution, and then transforming this to a target distribution using invertible neural networks. Currenlty there are two different such transforms available in Bijectors.jl: `PlanarFlow` and `RadialFlow`. Let's create a flow with a single `PlanarLayer`:

```julia
julia> d = MvNormal(zeros(10), ones(10))
       b = PlanarLayer(10)

PlanarLayer{Array{Float64,2},Array{Float64,1}}([-0.314585; -2.12398; … ; -1.03734; 0.358121], [-0.410737; -2.28809; … ; -0.973653; 0.498173], [1.43277])

julia> flow = transformed(d, b)
TransformedDistribution{MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Array{Float64,1}},PlanarLayer{Array{Float64,2},Array{Float64,1}},Multivariate}(
dist: DiagNormal(
dim: 10
μ: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Σ: [1.0 0.0 … 0.0 0.0; 0.0 1.0 … 0.0 0.0; … ; 0.0 0.0 … 1.0 0.0; 0.0 0.0 … 0.0 1.0]
)

transform: PlanarLayer{Array{Float64,2},Array{Float64,1}}([-0.314585; -2.12398; … ; -1.03734; 0.358121], [-0.410737; -2.28809; … ; -0.973653; 0.498173], [1.43277])
)


julia> flow isa MultivariateDistribution
true
```

That's it. Now we can sample from it using `rand` and compute the `logpdf`, like any other `Distribution`.

```julia
julia> y = rand(flow)
10-element Array{Float64,1}:
 -1.071445060874162  
 -0.5861125545541264 
 -0.20580680244677016
 -0.39722878482476176
 -1.389712016403798  
 -1.306541712448448  
 -0.30426655640478534
 -0.25236339697375954
 -0.8167239755371702 
 -0.2091496608178263 

julia> logpdf(flow, y)                        # uses inverse of `b`; not very efficient for `PlanarFlow` and not 100% accurate
-20.42265249674501

julia> x = rand(flow.dist)
10-element Array{Float64,1}:
 -1.3580971978613618  
 -0.14987419873885632 
 -0.43333586299952587 
  0.6873847861686496  
  0.4802780532275886  
  1.2128548836103794  
  1.231281336482941   
  0.5673980088490946  
  1.7291046521748124  
 -0.036465987288854064

julia> logpdf_forward(flow, x)                # more efficent and accurate
-15.14882873386875
```

Want to fit the flow?

```julia
julia> using Tracker

julia> b = PlanarLayer(10, param)                  # construct parameters using `param`
PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}}([-0.448776; 0.138082; … ; -0.203282; -0.054871] (tracked), [-1.70598; 0.714034; … ; 0.340427; 0.155935] (tracked), [-0.174542] (tracked))

julia> flow = transformed(d, b)
TransformedDistribution{MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Array{Float64,1}},PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}},Multivariate}(
dist: DiagNormal(
dim: 10
μ: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Σ: [1.0 0.0 … 0.0 0.0; 0.0 1.0 … 0.0 0.0; … ; 0.0 0.0 … 1.0 0.0; 0.0 0.0 … 0.0 1.0]
)

transform: PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}}([-0.448776; 0.138082; … ; -0.203282; -0.054871] (tracked), [-1.70598; 0.714034; … ; 0.340427; 0.155935] (tracked), [-0.174542] (tracked))
)

julia> rand(flow)
Tracked 10-element Array{Float64,1}:
 -0.43034610482235824
 -1.2291497791483754 
 -0.6669111894516624 
 -0.53669141546758   
  0.22203436167631224
  0.19375938341908486
 -0.7016018204533963 
  1.0353319678328805 
 -0.5932260118764895 
 -0.5127551044141234

julia> x = rand(flow.dist)

julia> Tracker.back!(logpdf_forward(flow, x), 1.0) # backprob

julia> Tracker.grad(b.w)
10×1 Array{Float64,2}:
  0.007970605904267344
 -0.00050931432618324 
 -0.012275698582014296
 -0.009238682005821862
  0.008625181060971085
 -0.009017315948208902
 -0.007548744417896309
  0.003679880842608043
 -0.004626588177230863
  0.003465272529470697
```

We can easily create more complex flows by simply doing `PlanarFlow(10) ∘ PlanarFlow(10) ∘ RadialFlow(10)` and so on.

Another useful function is the `forward(d::Distribution)` method. It is similar to `forward(b::Bijector)` in the sense that it does a forward pass of the entire process "sample then transform" and returns all the most useful quantities in process.

```julia
julia> x, y, logjac, logpdf_y = forward(flow) # sample + transform and returns all the useful quantities in one pass
(x = [-0.219525, -0.0934792, -0.314279, -0.165165, -1.62902, -0.32859, 0.514781, -0.3116, 0.957078, -0.069889], y = [0.0930996, 1.62717, -0.756388, 0.519735, -1.33456, 0.484004, 0.680621, -0.307233, 1.68176, -0.450381], logabsdetjac = 1.355433577912755, logpdf = -12.658641183050728)
```

This method is for example useful when computing quantities such as the _expected lower bound (ELBO)_ between this transformed distribution and some other joint density. If no analytical expression is available, we have to approximate the ELBO by a monte carlo estimate. But one term in the ELBO is the entropy of the base density, which we _do_ know analytically in this case. Using the analytical expression for the entropy and then using a monte carlo estimate for the rest of the terms in the ELBO gives an estimate with lower variance than if we used the monte carlo estimate for the entire expectation.

### Reference
#### `Bijector`
##### Methods
##### Bijectors

#### `TransformedDistribution`
