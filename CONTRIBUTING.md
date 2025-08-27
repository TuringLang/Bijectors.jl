# Contributing to Bijectors.jl

Thank you for your interest in contributing to Bijectors.jl! This guide will help you set up your development environment and understand how to run tests locally.

## Development Setup

 1. **Clone the repository**:
    
    ```bash
    git clone https://github.com/TuringLang/Bijectors.jl.git
    cd Bijectors.jl
    ```

 2. **Install dependencies**:
    
    ```bash
    julia --project=. -e "using Pkg; Pkg.instantiate()"
    ```
 3. **Verify the package loads**:
    
    ```bash
    julia --project=. -e "using Bijectors; println(\"Package loaded successfully\")"
    ```

## Running Tests

### Reproducing CI Test Failures

When CI tests fail, the output often shows incomplete reproduction instructions. Here are the correct commands to reproduce different CI scenarios locally:

#### Interface Tests

```bash
julia --project=. -e "ENV[\"GROUP\"] = \"Interface\"; using Pkg; Pkg.test()"
```

#### AD Tests with Specific Backend

Replace `BackendName` with the failing AD backend from the CI output:

```bash
# ForwardDiff
julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"ForwardDiff\"; using Pkg; Pkg.test()"

# ReverseDiff  
julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"ReverseDiff\"; using Pkg; Pkg.test()"

# Tracker
julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"Tracker\"; using Pkg; Pkg.test()"

# Enzyme (may fail on some systems - this is expected)
julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"Enzyme\"; using Pkg; Pkg.test()"

# Mooncake
julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"Mooncake\"; using Pkg; Pkg.test()"
```

#### Full Test Suite

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

### CI Test Matrix

Our CI runs tests with the following configurations:

  - **Interface tests**: Core functionality tests (GROUP=Interface)
  - **AD tests**: Automatic differentiation tests with multiple backends (GROUP=AD, AD={ForwardDiff,ReverseDiff,Tracker,Enzyme,Mooncake})

### Important Notes

 1. **Always use `--project=.`**: This ensures Julia uses the correct project environment with the right dependencies.

 2. **Environment variables matter**: The `GROUP` and `AD` environment variables control which tests run, matching the CI configuration.
 3. **Enzyme tests may fail**: Due to system compatibility issues, Enzyme tests may fail on some machines. This is expected and not a blocker for development.
 4. **Test timing**:
    
      + Interface tests: ~7.5 minutes
      + AD tests (single backend): ~1.5 minutes
      + Full test suite: 20+ minutes

## Code Formatting

We use JuliaFormatter.jl for consistent code formatting:

```bash
# Install formatter (one-time setup)
julia --project=. -e "using Pkg; Pkg.add(\"JuliaFormatter\")"

# Format code
julia --project=. -e "using JuliaFormatter; format(\".\")"
```

## Documentation

To build documentation locally:

```bash
julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include(\"docs/make.jl\")"
```

## Validation Checklist

Before submitting a PR, please ensure:

  - [ ] Package loads without errors: `julia --project=. -e "using Bijectors"`
  - [ ] Interface tests pass: `julia --project=. -e "ENV[\"GROUP\"] = \"Interface\"; using Pkg; Pkg.test()"`
  - [ ] Code is formatted: `julia --project=. -e "using JuliaFormatter; format(\".\")"`
  - [ ] Basic functionality works (see test scenarios below)

### Basic Test Scenarios

You can manually verify basic functionality:

```julia
using Bijectors, Distributions

# Test 1: Basic bijector for LogNormal  
d = LogNormal()
b = bijector(d)
x = 1.0
y = b(x)  # Should return 0.0

# Test 2: Inverse transformation
x_reconstructed = inverse(b)(y)  # Should return 1.0

# Test 3: Log absolute determinant of Jacobian
logjac = logabsdetjac(b, x)

# Test 4: Combined transformation  
z, logabsdet = with_logabsdet_jacobian(b, x)
```

## Getting Help

If you need help:

  - Join the #turing channel on [Julia Slack](https://julialang.org/slack/)
  - Ask questions on [Julia Discourse](https://discourse.julialang.org)
  - Open a GitHub issue for bugs or feature requests

Thank you for contributing to Bijectors.jl!
