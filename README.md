# Bijectors.jl

[![Docs - Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/Bijectors.jl/stable)
[![Docs - Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/Bijectors.jl/dev)
[![Interface tests](https://github.com/TuringLang/Bijectors.jl/workflows/Interface%20tests/badge.svg?branch=main)](https://github.com/TuringLang/Bijectors.jl/actions?query=workflow%3A%22Interface+tests%22+branch%3Amain)
[![AD tests](https://github.com/TuringLang/Bijectors.jl/workflows/AD%20tests/badge.svg?branch=main)](https://github.com/TuringLang/Bijectors.jl/actions?query=workflow%3A%22AD+tests%22+branch%3Amain)

*A package for transforming distributions, used by [Turing.jl](https://github.com/TuringLang/Turing.jl).*

Bijectors.jl implements both an interface for transforming distributions from Distributions.jl and many transformations needed in this context.
This package is used heavily in the probabilistic programming language Turing.jl.

See the [documentation](https://turinglang.github.io/Bijectors.jl) for more.

## Development

### Running Tests Locally

To reproduce CI test failures locally, use the following commands:

```bash
# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Interface tests (reproduces "Interface tests" CI workflow)
julia --project=. -e "ENV[\"GROUP\"] = \"Interface\"; using Pkg; Pkg.test()"

# AD tests with specific backend (reproduces "AD tests" CI workflow)
# Replace "ReverseDiff" with the specific AD backend from the failing CI job
julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"ReverseDiff\"; using Pkg; Pkg.test()"

# Other AD backends available:
# julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"ForwardDiff\"; using Pkg; Pkg.test()"
# julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"Mooncake\"; using Pkg; Pkg.test()"
# julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"Tracker\"; using Pkg; Pkg.test()"
# julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"Enzyme\"; using Pkg; Pkg.test()"

# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"
```

**Note**: When reproducing CI failures, ensure you use the correct environment variables (`GROUP` and `AD`) and always include `--project=.` in your Julia commands. These are required for the tests to run in the same configuration as CI.

For more detailed development instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Do you want to contribute?

If you feel you have some relevant skills and are interested in contributing, please get in touch!
You can find us in the #turing channel on the [Julia Slack](https://julialang.org/slack/) or [Discourse](https://discourse.julialang.org).
If you're having any problems, please open a Github issue, even if the problem seems small (like help figuring out an error message).
Every issue you open helps us to improve the library!
