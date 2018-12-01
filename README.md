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

1. `link`: maps a sample of a random distribution `dist` from its support to a value in [`-Inf`, `Inf`]. Example:

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
