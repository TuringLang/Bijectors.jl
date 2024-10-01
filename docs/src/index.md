# Bijectors.jl

This package implements a set of functions for transforming constrained random variables (e.g. simplexes, intervals) to Euclidean space. The 3 main functions implemented in this package are the `link`, `invlink` and `logpdf_with_trans` for a number of distributions. The distributions supported are:

 1. `RealDistribution`: `Union{Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal, NormalCanon, TDist}`,
 2. `PositiveDistribution`: `Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull}`,
 3. `UnitDistribution`: `Union{Beta, KSOneSided, NoncentralBeta}`,
 4. `SimplexDistribution`: `Union{Dirichlet}`,
 5. `PDMatDistribution`: `Union{InverseWishart, Wishart}`, and
 6. `TransformDistribution`: `Union{T, Truncated{T}} where T<:ContinuousUnivariateDistribution`.

All exported names from the [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) package are reexported from `Bijectors`.

Bijectors.jl also provides a nice interface for working with these maps: composition, inversion, etc.
The following table lists mathematical operations for a bijector and the corresponding code in Bijectors.jl.

| Operation                                   | Method                          | Automatic |
|:-------------------------------------------:|:-------------------------------:|:---------:|
| `b ↦ b⁻¹`                                   | `inverse(b)`                    | ✓         |
| `(b₁, b₂) ↦ (b₁ ∘ b₂)`                      | `b₁ ∘ b₂`                       | ✓         |
| `(b₁, b₂) ↦ [b₁, b₂]`                       | `stack(b₁, b₂)`                 | ✓         |
| `x ↦ b(x)`                                  | `b(x)`                          | ×         |
| `y ↦ b⁻¹(y)`                                | `inverse(b)(y)`                 | ×         |
| `x ↦ log｜det J(b, x)｜`                      | `logabsdetjac(b, x)`            | AD        |
| `x ↦ b(x), log｜det J(b, x)｜`                | `with_logabsdet_jacobian(b, x)` | ✓         |
| `p ↦ q := b_* p`                            | `q = transformed(p, b)`         | ✓         |
| `y ∼ q`                                     | `y = rand(q)`                   | ✓         |
| `p ↦ b` such that `support(b_* p) = ℝᵈ`     | `bijector(p)`                   | ✓         |
| `(x ∼ p, b(x), log｜det J(b, x)｜, log q(y))` | `forward(q)`                    | ✓         |

In this table, `b` denotes a `Bijector`, `J(b, x)` denotes the Jacobian of `b` evaluated at `x`, `b_*` denotes the [push-forward](https://www.wikiwand.com/en/Pushforward_measure) of `p` by `b`, and `x ∼ p` denotes `x` sampled from the distribution with density `p`.

The "Automatic" column in the table refers to whether or not you are required to implement the feature for a custom `Bijector`. "AD" refers to the fact that it can be implemented "automatically" using automatic differentiation.
