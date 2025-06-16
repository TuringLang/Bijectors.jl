# Bijectors.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/Bijectors.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/Bijectors.jl/dev)
[![Interface tests](https://github.com/TuringLang/Bijectors.jl/workflows/Interface%20tests/badge.svg?branch=main)](https://github.com/TuringLang/Bijectors.jl/actions?query=workflow%3A%22Interface+tests%22+branch%3Amain)
[![AD tests](https://github.com/TuringLang/Bijectors.jl/workflows/AD%20tests/badge.svg?branch=main)](https://github.com/TuringLang/Bijectors.jl/actions?query=workflow%3A%22AD+tests%22+branch%3Amain)

*A package for transforming distributions, used by [Turing.jl](https://github.com/TuringLang/Turing.jl).*

Bijectors.jl implements both an interface for transforming distributions from Distributions.jl and many transformations needed in this context. This package is used heavily in the probabilistic programing language Turing.jl.

## Installation

Bijectors.jl is a registered Julia package. You can install it using the package manager:

```julia
using Pkg
Pkg.add("Bijectors")
```

Or in the Julia REPL package mode (press `]`):

```
pkg> add Bijectors
```

## Quick Start

Here's a simple example of using a bijector to transform a distribution:

```julia
using Bijectors, Distributions

# Create a normal distribution
d = Normal(0, 1)

# Apply a log transformation (maps real numbers to positive reals)
b = Log()
transformed_d = transformed(d, b)

# Sample from the transformed distribution
x = rand(transformed_d, 100)

# The log-likelihood accounts for the transformation
logpdf(transformed_d, x[1])
```

## Key Concepts

- **Bijector**: A differentiable, invertible transformation with computable Jacobian
- **Transformed distributions**: Apply bijectors to change the support of distributions
- **Common transformations**: Log (for positive reals), Logit (for unit interval), Simplex (for probability simplices)

See the [documentation](https://turinglang.github.io/Bijectors.jl) for comprehensive guides and API reference.

## Development

### Local Setup

To set up the development environment:

```julia
# Clone and navigate to the repository
# julia --project=.

# Install dependencies
using Pkg
Pkg.instantiate()
```

### Testing

Run the full test suite:

```julia
# All tests
julia --project=. -e "using Pkg; Pkg.test()"

# Specific test groups
GROUP=Interface julia --project=. -e "using Pkg; Pkg.test()"
GROUP=AD julia --project=. -e "using Pkg; Pkg.test()"
```

### Documentation

Build documentation locally:

```julia
julia --project=docs docs/make.jl
```

## Do you want to contribute?

If you feel you have some relevant skills and are interested in contributing, please get in touch! You can find us in the #turing channel on the [Julia Slack](https://julialang.org/slack/) or [Discourse](https://discourse.julialang.org). If you're having any problems, please open a Github issue, even if the problem seems small (like help figuring out an error message). Every issue you open helps us to improve the library!
