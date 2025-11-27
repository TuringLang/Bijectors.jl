<p align="center">
<h1>Bijectors.jl</h1>
<a href="https://turinglang.github.io/Bijectors.jl"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Documentation for latest stable release"></a>
<a href="https://turinglang.github.io/Bijectors.jl/dev"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation for development version"></a>
<a href="https://github.com/TuringLang/Bijectors.jl/actions/workflows/CI.yml"><img src="https://github.com/TuringLang/Bijectors.jl/actions/workflows/CI.yml/badge.svg" alt="CI"></a>
</p>

Bijectors.jl implements functions for transforming random variables and probability distributions.

A quick overview of some of the key functionality is provided below:

```julia
julia> using Bijectors;
       dist = LogNormal();
LogNormal{Float64}(μ=0.0, σ=1.0)

julia> x = rand(dist)      # Constrained to (0, ∞)
0.6471106974390148

julia> b = bijector(dist)  # This maps from (0, ∞) to ℝ
(::Base.Fix1{typeof(broadcast), typeof(log)}) (generic function with 1 method)

julia> y = b(x)            # Unconstrained value in ℝ
-0.43523790570180304

julia> # Log-absolute determinant of the Jacobian at x.
       with_logabsdet_jacobian(b, x)
(-0.43523790570180304, 0.43523790570180304)
```

Please see the [documentation](https://turinglang.github.io/Bijectors.jl) for more information.

## Get in touch

If you have any questions, please feel free to [post on Julia Slack](https://julialang.slack.com/archives/CCYDC34A0) or [Discourse](https://discourse.julialang.org/).
We also very much welcome GitHub issues or pull requests!
