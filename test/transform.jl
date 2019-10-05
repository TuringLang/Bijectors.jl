using Test
using Bijectors
using ForwardDiff: derivative, jacobian
using LinearAlgebra: logabsdet, I, norm
using Random

Random.seed!(123)

# logabsdet doesn't handle scalars.
_logabsdet(x::AbstractArray) = logabsdet(x)[1]
_logabsdet(x::Real) = log(abs(x))

# Generate a (vector / matrix of) random number(s).
_rand_real(::Real) = randn()
_rand_real(x) = randn(size(x))
_rand_real(x, e) = (y = randn(size(x)); y[end] = e; y)

# Standard tests for all distributions involving a single-sample.
function single_sample_tests(dist, jacobian)
    ϵ = eps(Float64)
    # Do the regular single-sample tests.
    single_sample_tests(dist)

    # Check that the implementation of the logpdf agrees with the AD version.
    x = rand(dist)
    if dist isa SimplexDistribution
        logpdf_ad = logpdf(dist, x .+ ϵ) - _logabsdet(jacobian(x->link(dist, x, Val{false}), x))
    else
        logpdf_ad = logpdf(dist, x) - _logabsdet(jacobian(x->link(dist, x), x))
    end
    @test logpdf_ad ≈ logpdf_with_trans(dist, x, true)
end

# Standard tests for all distributions involving a single-sample. Doesn't check that the
# logpdf implementation is consistent with the link function for technical reasons.
function single_sample_tests(dist)
    ϵ = eps(Float64)

    # Check that invlink is inverse of link.
    x = rand(dist)
    @test invlink(dist, link(dist, copy(x))) ≈ x atol=1e-9

    # Check that link is inverse of invlink. Hopefully this just holds given the above...
    y = link(dist, x)
    if dist isa Dirichlet
        # `logit` and `logistic` are not perfect inverses. This leads to a diversion.
        # Example:
        # julia> logit(logistic(0.9999999999999998))
        #    1.0
        # julia> logistic(logit(0.9999999999999998))
        # 0.9999999999999998
        @test link(dist, invlink(dist, copy(y))) ≈ y atol=0.5
    else
        @test link(dist, invlink(dist, copy(y))) ≈ y atol=1e-9
    end
    if dist isa SimplexDistribution
        # This should probably be exact.
        @test logpdf(dist, x .+ ϵ) == logpdf_with_trans(dist, x, false)
        # Check that invlink maps back to the apppropriate constrained domain.
        @test all(isfinite, logpdf.(Ref(dist), [invlink(dist, _rand_real(x, 0)) .+ ϵ for _ in 1:100]))
    else
        # This should probably be exact.
        @test logpdf(dist, x) == logpdf_with_trans(dist, x, false)
        @test all(isfinite, logpdf.(Ref(dist), [invlink(dist, _rand_real(x)) for _ in 1:100]))
    end
    # This is a quirk of the current implementation, of which it would be nice to be rid.
    @test typeof(x) == typeof(y)
end

# Standard tests for all distributions involving multiple samples. xs should be whatever
# the appropriate repeated version of x is for the distribution in question. ie. for
# univariate distributions, just a vector of identical values. For vector-valued
# distributions, a matrix whose columns are identical.
function multi_sample_tests(dist, x, xs, N)
    ys = link(dist, copy(xs))
    @test invlink(dist, link(dist, copy(xs))) ≈ xs atol=1e-9
    @test link(dist, invlink(dist, copy(ys))) ≈ ys atol=1e-9
    @test logpdf_with_trans(dist, xs, true) == fill(logpdf_with_trans(dist, x, true), N)
    @test logpdf_with_trans(dist, xs, false) == fill(logpdf_with_trans(dist, x, false), N)

    # This is a quirk of the current implementation, of which it would be nice to be rid.
    @test typeof(xs) == typeof(ys)
end

# Scalar tests
@testset "scalar" begin
let
    # Tests with scalar-valued distributions.
    uni_dists = [
        Arcsine(2, 4),
        Beta(2,2),
        BetaPrime(),
        Biweight(),
        Cauchy(),
        Chi(3),
        Chisq(2),
        Cosine(),
        Epanechnikov(),
        Erlang(),
        Exponential(),
        FDist(1, 1),
        Frechet(),
        Gamma(),
        InverseGamma(),
        InverseGaussian(),
        # Kolmogorov(),
        Laplace(),
        Levy(),
        Logistic(),
        LogNormal(1.0, 2.5),
        Normal(0.1, 2.5),
        Pareto(),
        Rayleigh(1.0),
        TDist(2),
        TruncatedNormal(0, 1, -Inf, 2),
    ]
    for dist in uni_dists

        single_sample_tests(dist, derivative)

        # specialised multi-sample tests.
        N = 10
        x = rand(dist)
        xs = fill(x, N)
        multi_sample_tests(dist, x, xs, N)
    end
end
end

# Tests with vector-valued distributions.
@testset "vector" begin
let ϵ = eps(Float64)
    vector_dists = [
        Dirichlet(2, 3),
        Dirichlet([1000 * one(Float64), eps(Float64)]),
        Dirichlet([eps(Float64), 1000 * one(Float64)]),
        MvNormal(randn(10), exp.(randn(10))),
        MvLogNormal(MvNormal(randn(10), exp.(randn(10)))),
        Dirichlet([1000 * one(Float64), eps(Float64)]), 
        Dirichlet([eps(Float64), 1000 * one(Float64)]),
    ]
    for dist in vector_dists

        if dist isa Dirichlet
            single_sample_tests(dist)

            # This should fail at the minute. Not sure what the correct way to test this is.
            x = rand(dist)
            logpdf_turing = logpdf_with_trans(dist, x, true)
            J = jacobian(x->link(dist, x, Val{false}), x)
            @test logpdf(dist, x .+ ϵ) - _logabsdet(J) ≈ logpdf_turing

            # Issue #12
            stepsize = 1e10
            dim = length(dist)
            x = [logpdf_with_trans(dist, invlink(dist, link(dist, rand(dist)) .+ randn(dim) .* stepsize), true) for _ in 1:1_000]
            @test !any(isinf, x) && !any(isnan, x)
        else
            single_sample_tests(dist, jacobian)
        end

        # Multi-sample tests. Columns are observations due to Distributions.jl conventions.
        N = 10
        x = rand(dist)
        xs = repeat(x, 1, N)
        multi_sample_tests(dist, x, xs, N)
    end
end
end

# Tests with matrix-valued distributions.
@testset "matrix" begin
let
    matrix_dists = [
        Wishart(7, [1 0.5; 0.5 1]),
        InverseWishart(2, [1 0.5; 0.5 1]),
    ]
    for dist in matrix_dists

        single_sample_tests(dist)

        x = rand(dist); x = x + x' + 2I
        lowerinds = [LinearIndices(size(x))[I] for I in CartesianIndices(size(x)) if I[1] >= I[2]]
        upperinds = [LinearIndices(size(x))[I] for I in CartesianIndices(size(x)) if I[2] >= I[1]]
        logpdf_turing = logpdf_with_trans(dist, x, true)
        J = jacobian(x->link(dist, x), x)
        J = J[lowerinds, upperinds]
        @test logpdf(dist, x) - _logabsdet(J) ≈ logpdf_turing

        # Multi-sample tests comprising vectors of matrices.
        N = 10
        x = rand(dist)
        xs = [x for _ in 1:N]
        multi_sample_tests(dist, x, xs, N)
    end
end
end

################################## Miscelaneous old tests ##################################

# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), exp.([-1000., -1000., -1000.]), true)
# NaN
# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), [-1000., -1000., -1000.], true, true)
# -1999.30685281944
#
# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), exp.([-1., -2., -3.]), true)
# -3.006450206744678
# julia> logpdf_with_trans(Dirichlet([1., 1., 1.]), [-1., -2., -3.], true, true)
# -3.006450206744678
d  = Dirichlet([1., 1., 1.])
r  = [-1000., -1000., 0.0]
r2 = [-1., -2., 0.0]

# test link
#link(d, r)

# test invlink
@test invlink(d, r) ≈ [0., 0., 1.] atol=1e-9

# test logpdf_with_trans
#@test logpdf_with_trans(d, invlink(d, r), true) -1999.30685281944 1e-9 ≈ # atol=NaN
@test logpdf_with_trans(d, invlink(d, r2), true) ≈ -3.760398892580863 atol=1e-9

macro aeq(x, y)
    return quote
        x = $(esc(x))
        y = $(esc(y))
        norm = $(esc(:norm))
        norm(x - y) <= 1e-10
    end
end

@testset "Dirichlet Jacobians" begin
    function test_link_and_invlink()
        dist = Dirichlet(4, 4)
        x = rand(dist)
        y = link(dist, x)

        f1 = x -> link(dist, x, Val{true})
        f2 = x -> link(dist, x, Val{false})
        g1 = y -> invlink(dist, y, Val{true})
        g2 = y -> invlink(dist, y, Val{false})

        @test @aeq jacobian(f1, x) Bijectors.link_jacobian(dist, x, Val{true})
        @test @aeq jacobian(f2, x) Bijectors.link_jacobian(dist, x, Val{false})
        @test @aeq jacobian(g1, y) Bijectors.invlink_jacobian(dist, y, Val{true})
        @test @aeq jacobian(g2, y) Bijectors.invlink_jacobian(dist, y, Val{false})
        @test @aeq Bijectors.link_jacobian(dist, x, Val{false}) * Bijectors.invlink_jacobian(dist, y, Val{false}) I
    end
    test_link_and_invlink()
end
