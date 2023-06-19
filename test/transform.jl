using Test
using Bijectors
using ForwardDiff: ForwardDiff
using LinearAlgebra: logabsdet, I, norm
using Random

Random.seed!(123)

# logabsdet doesn't handle scalars.
_logabsdet(x::AbstractArray) = logabsdet(x)[1]
_logabsdet(x::Real) = log(abs(x))

# Generate a (vector / matrix of) random number(s).
_rand_real(dist, ::Real) = randn()
function _rand_real(dist, x)
    b = bijector(dist)
    sz = Bijectors.output_size(b, size(x))
    return randn(sz)
end

# Standard tests for all distributions involving a single-sample.
function single_sample_tests(dist, jacobian)
    ϵ = eps(Float64)
    # Do the regular single-sample tests.
    single_sample_tests(dist)

    # Check that the implementation of the logpdf agrees with the AD version.
    x = rand(dist)
    if dist isa SimplexDistribution
        logpdf_ad =
            logpdf(dist, x .+ ϵ) - _logabsdet(jacobian(x -> link(dist, x, false), x))
    else
        logpdf_ad = logpdf(dist, x) - _logabsdet(jacobian(x -> link(dist, x), x))
    end
    @test logpdf_ad ≈ logpdf_with_trans(dist, x, true)
end

# Standard tests for all distributions involving a single-sample. Doesn't check that the
# logpdf implementation is consistent with the link function for technical reasons.
function single_sample_tests(dist)
    ϵ = eps(Float64)

    # Check that invlink is inverse of link.
    x = rand(dist)

    if dist isa LKJCholesky
        x_inv = @inferred Cholesky{Float64,Matrix{Float64}} invlink(
            dist, link(dist, copy(x))
        )
        @test x_inv.UL ≈ x.UL atol = 1e-9
    else
        @test @inferred(invlink(dist, link(dist, copy(x)))) ≈ x atol = 1e-9
    end

    # Check that link is inverse of invlink. Hopefully this just holds given the above...
    y = @inferred(link(dist, x))
    if dist isa Dirichlet
        # `logit` and `logistic` are not perfect inverses. This leads to a diversion.
        # Example:
        # julia> logit(logistic(0.9999999999999998))
        #    1.0
        # julia> logistic(logit(0.9999999999999998))
        # 0.9999999999999998
        @test @inferred(link(dist, invlink(dist, copy(y)))) ≈ y atol = 0.5
    else
        @test @inferred(link(dist, invlink(dist, copy(y)))) ≈ y atol = 1e-9
    end
    if dist isa SimplexDistribution
        # This should probably be exact.
        @test logpdf(dist, x .+ ϵ) == logpdf_with_trans(dist, x, false)
        # Check that invlink maps back to the apppropriate constrained domain.
        @test all(
            isfinite,
            logpdf.(Ref(dist), [invlink(dist, _rand_real(dist, x)) .+ ϵ for _ in 1:100]),
        )
    else
        # This should probably be exact.
        @test logpdf(dist, x) == logpdf_with_trans(dist, x, false)
        @test all(
            isfinite,
            logpdf.(Ref(dist), [invlink(dist, _rand_real(dist, y)) for _ in 1:100]),
        )
    end
end

# Scalar tests
@testset "scalar" begin
    let
        # Tests with scalar-valued distributions.
        uni_dists = [
            Arcsine(2, 4),
            Beta(2, 2),
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
            truncated(Normal(0, 1), -Inf, 2),
        ]
        for dist in uni_dists
            single_sample_tests(dist, ForwardDiff.derivative)
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
            MvNormal(randn(10), Diagonal(exp.(randn(10)))),
            MvLogNormal(MvNormal(randn(10), Diagonal(exp.(randn(10))))),
            Dirichlet([1000 * one(Float64), eps(Float64)]),
            Dirichlet([eps(Float64), 1000 * one(Float64)]),
        ]
        for dist in vector_dists
            if dist isa Dirichlet
                single_sample_tests(dist)

                # This should fail at the minute. Not sure what the correct way to test this is.

                # Workaround for intermittent test failures, result of `logpdf_with_trans(dist, x, true)`
                # is incorrect for `x == [0.9999999999999998, 0.0]`:
                x =
                    if params(dist) ==
                        params(Dirichlet([1000 * one(Float64), eps(Float64)]))
                        [1.0, 0.0]
                    else
                        rand(dist)
                    end
                # `Dirichlet` is no longer mapping between spaces of the same dimensionality,
                # so the block below no longer works.
                if !(dist isa Dirichlet)
                    logpdf_turing = logpdf_with_trans(dist, x, true)
                    J = ForwardDiff.jacobian(x -> link(dist, x), x)
                    @test logpdf(dist, x .+ ϵ) - _logabsdet(J) ≈ logpdf_turing
                end

                # Issue #12
                stepsize = 1e10
                dim = Bijectors.output_length(bijector(dist), length(dist))
                x = [
                    logpdf_with_trans(
                        dist,
                        invlink(dist, link(dist, rand(dist)) .+ randn(dim) .* stepsize),
                        true,
                    ) for _ in 1:1_000
                ]
                @test !any(isinf, x) && !any(isnan, x)
            else
                single_sample_tests(dist, ForwardDiff.jacobian)
            end
        end
    end
end

# Tests with matrix-valued distributions.
@testset "matrix" begin
    let
        matrix_dists = [Wishart(7, [1 0.5; 0.5 1]), InverseWishart(2, [1 0.5; 0.5 1])]
        for dist in matrix_dists
            single_sample_tests(dist)

            x = rand(dist)
            x = x + x' + 2I
            upperinds = [
                LinearIndices(size(x))[I] for I in CartesianIndices(size(x)) if I[2] >= I[1]
            ]
            logpdf_turing = logpdf_with_trans(dist, x, true)
            J = ForwardDiff.jacobian(x -> link(dist, x), x)
            J = J[:, upperinds]
            @test logpdf(dist, x) - _logabsdet(J) ≈ logpdf_turing
        end
    end
end

@testset "LKJ" begin
    dist = LKJ(3, 1)

    single_sample_tests(dist)

    x = rand(dist)
    x = x + x' + 2I
    d = 1 ./ sqrt.(diag(x))
    x = d .* x .* d'

    upperinds = [
        LinearIndices(size(x))[I] for I in CartesianIndices(size(x)) if I[2] > I[1]
    ]
    J = ForwardDiff.jacobian(x -> link(dist, x), x)
    J = J[:, upperinds]
    logpdf_turing = logpdf_with_trans(dist, x, true)
    @test logpdf(dist, x) - _logabsdet(J) ≈ logpdf_turing
end

@testset "LKJCholesky" begin
    dist = LKJCholesky(3, 1)

    single_sample_tests(dist)

    x = rand(dist)

    upperinds = [
        LinearIndices(size(x))[I] for I in CartesianIndices(size(x)) if I[2] > I[1]
    ]
    J = ForwardDiff.jacobian(x -> link(dist, x), x.U)
    J = J[:, upperinds]
    logpdf_turing = logpdf_with_trans(dist, x, true)
    @test logpdf(dist, x) - _logabsdet(J) ≈ logpdf_turing
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
d = Dirichlet([1.0, 1.0, 1.0])
r = [-1000.0, -1000.0]
r2 = [-1.0, -2.0]

# test vector invlink
dist = Dirichlet(ones(5))
x = [[-2.72689, -2.92751, 1.63114, -1.62054, 0.0] [
    -1.24249, 2.58902, -3.73043, -3.53685, 0.0
]]
@test all(sum(Bijectors.invlink(dist, x); dims=1) .== 1)

# test link
#link(d, r)

# test invlink
@test invlink(d, r) ≈ [0.0, 0.0, 1.0] atol = 1e-9

# test logpdf_with_trans
#@test logpdf_with_trans(d, invlink(d, r), true) -1999.30685281944 1e-9 ≈ # atol=NaN
@test logpdf_with_trans(d, invlink(d, r2), true) ≈ -3.760398892580863 atol = 1e-9

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
        y = @inferred(link(dist, x))

        f1 = x -> link(dist, x)
        g1 = y -> invlink(dist, y)

        @test @aeq ForwardDiff.jacobian(f1, x) @inferred(Bijectors.simplex_link_jacobian(x))
        @test @aeq ForwardDiff.jacobian(g1, y) @inferred(
            Bijectors.simplex_invlink_jacobian(y)
        )
        @test @aeq Bijectors.simplex_link_jacobian(x) *
            Bijectors.simplex_invlink_jacobian(y) I
    end
    for i in 1:4
        test_link_and_invlink()
    end
end
