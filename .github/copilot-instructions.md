# Bijectors.jl Development Instructions

**ALWAYS** follow these instructions and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.

## Overview

Bijectors.jl is a Julia package for transforming probability distributions, implementing both an interface for transforming distributions from Distributions.jl and many transformations needed in this context. Used heavily in the probabilistic programming language Turing.jl.

## Working Effectively

### Quick Setup and Build

```bash
# Install dependencies (NEVER CANCEL: takes ~30 seconds)
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Test package loading
julia --project=. -e "using Bijectors; println(\"Package loaded successfully\")"
```

### Running Tests

```bash
# Interface tests only (NEVER CANCEL: takes ~7.5 minutes, timeout 30+ minutes)
julia --project=. -e "ENV[\"GROUP\"] = \"Interface\"; using Pkg; Pkg.test()"

# AD tests with specific backend (NEVER CANCEL: takes ~1.5 minutes, timeout 10+ minutes)
julia --project=. -e "ENV[\"GROUP\"] = \"AD\"; ENV[\"AD\"] = \"ForwardDiff\"; using Pkg; Pkg.test()"

# Full test suite (NEVER CANCEL: may take 20+ minutes and some Enzyme tests may fail - this is expected)
julia --project=. -e "using Pkg; Pkg.test()"
```

**CRITICAL**:

  - **NEVER CANCEL** any test commands - they can take significant time to complete
  - Set timeouts to 30+ minutes for Interface tests, 60+ minutes for full test suite
  - Enzyme tests may fail due to system compatibility issues - this is expected and not a blocker

### Documentation

```bash
# Build documentation (NEVER CANCEL: takes ~47 seconds, timeout 5+ minutes)
julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include(\"docs/make.jl\")"
```

### Code Formatting

```bash
# Install formatter (one-time setup, takes ~45 seconds)
julia --project=. -e "using Pkg; Pkg.add(\"JuliaFormatter\")"

# Format code
julia --project=. -e "using JuliaFormatter; format(\".\")"
```

## Validation Scenarios

**ALWAYS** test basic functionality after making changes:

```julia
# Test basic bijector operations
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

## Test Groups and Timing Expectations

  - **Interface tests**: ~7.5 minutes - Core functionality tests
  - **AD tests (single backend)**: ~1.5 minutes - Automatic differentiation tests
  - **Full test suite**: 20+ minutes - All tests including multiple AD backends
  - **Documentation build**: ~47 seconds - Generate documentation
  - **Package installation**: ~27 seconds - Install dependencies

## CI and Development Workflow

### Before Committing Changes

 1. **ALWAYS** run Interface tests: `ENV["GROUP"] = "Interface"; Pkg.test()`
 2. **ALWAYS** run formatting: `using JuliaFormatter; format(".")`
 3. Test basic functionality scenarios listed above
 4. If changing AD-related code, run specific AD backend tests

### CI Workflows

  - **Interface tests**: Core functionality (runs on PRs)
  - **AD tests**: Multiple AD backends including ForwardDiff, ReverseDiff, Tracker, Enzyme, Mooncake
  - **Format check**: Uses TuringLang/actions/Format
  - **Documentation**: Builds and deploys docs

## Common Issues and Workarounds

  - **Enzyme tests failing**: Expected on some systems - not a blocker for development
  - **Network issues during package installation**: May see "could not download" warnings but installation continues
  - **Documentation git warnings**: Expected in sandboxed environments
  - **Deprecation warnings**: Some warnings about MatrixReshaped are expected

## Key Repository Structure

```
.
├── Project.toml              # Package definition and dependencies
├── src/                      # Main source code
│   ├── Bijectors.jl         # Main module
│   ├── interface.jl         # Core interface definitions
│   ├── chainrules.jl        # AD integration
│   └── bijectors/           # Individual bijector implementations
├── test/                     # Test suite
│   ├── runtests.jl          # Main test runner
│   ├── ad/                  # AD-specific tests
│   └── bijectors/           # Bijector-specific tests
├── docs/                     # Documentation
├── ext/                      # Package extensions for AD backends
└── .github/workflows/        # CI configuration
```

## Dependencies and Extensions

Core dependencies include Distributions.jl, ChainRulesCore, and various math packages. Extensions provide integration with AD backends:

  - ForwardDiff, ReverseDiff, Tracker, Enzyme, Mooncake
  - DistributionsAD, LazyArrays

## Manual Validation Requirements

After any changes:

 1. Build the package successfully
 2. Run Interface tests (minimum requirement)
 3. Test at least one complete bijector transformation scenario
 4. Verify inverse transformations work correctly
 5. Check log absolute determinant Jacobian calculations
 6. Run formatter before committing

The package should load without errors and basic transformations should work as demonstrated in the validation scenarios above.
