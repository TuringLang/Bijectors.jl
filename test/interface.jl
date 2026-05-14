using Test
using Random
using LinearAlgebra
using ForwardDiff

using Bijectors
Random.seed!(123)

@testset "Univariate" begin
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
        transformed(Beta(2, 2)),
        transformed(Exponential()),
    ]

    for dist in uni_dists
        @testset "$dist: dist" begin
            td = @inferred transformed(dist)

            # single sample
            y = @inferred rand(td)
            x = @inferred inverse(td.transform)(y)
            @test y ≈ @inferred td.transform(x)
            @test @inferred(logpdf(td, y)) ≈ @inferred(logpdf_with_trans(dist, x, true))

            # multi-sample
            y = @inferred rand(td, 10)
            x = inverse(td.transform).(y)
            @test logpdf.(td, y) ≈ logpdf_with_trans.(dist, x, true)

            # logpdf corresponds to logpdf_with_trans
            d = dist
            b = @inferred bijector(d)
            x = rand(d)
            y = @inferred b(x)
            @test logpdf(d, inverse(b)(y)) + logabsdetjacinv(b, y) ≈
                logpdf_with_trans(d, x, true)
            @test logpdf(d, x) - logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, true)

            # verify against AD
            d = dist
            b = bijector(d)
            x = rand(d)
            y = b(x)
            # `ForwardDiff.derivative` can lead to some numerical inaccuracy,
            # so we use a slightly higher `atol` than default.
            @test log(abs(ForwardDiff.derivative(b, x))) ≈ logabsdetjac(b, x) atol = 1e-6
            @test log(abs(ForwardDiff.derivative(inverse(b), y))) ≈
                logabsdetjac(inverse(b), y) atol = 1e-6
        end
    end

    @testset "logabsdetjac numerical stability: Bijectors.jl#325" begin
        d = Uniform(-1, 1)
        b = bijector(d)
        y = 80
        # x needs higher precision to be calculated correctly, otherwise
        # logpdf_with_trans returns -Inf
        d_big = Uniform(big(-1.0), big(1.0))
        b_big = bijector(d_big)
        x_big = inverse(b_big)(big(y))
        @test logpdf(d_big, x_big) + logabsdetjacinv(b, y) ≈
            logpdf_with_trans(d_big, x_big, true) atol = 1e-14
        @test logpdf(d_big, x_big) - logabsdetjac(b, x_big) ≈
            logpdf_with_trans(d_big, x_big, true) atol = 1e-14
    end
end

@testset "Truncated" begin
    d = truncated(Normal(), -1, 1)
    b = bijector(d)
    x = rand(d)
    y = b(x)
    @test y ≈ link(d, x)
    @test inverse(b)(y) ≈ x
    @test logabsdetjac(b, x) ≈
        logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)

    d = truncated(Normal(), -Inf, 1)
    b = bijector(d)
    x = rand(d)
    y = b(x)
    @test y ≈ link(d, x)
    @test inverse(b)(y) ≈ x
    @test logabsdetjac(b, x) ≈
        logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)

    d = truncated(Normal(), 1, Inf)
    b = bijector(d)
    x = rand(d)
    y = b(x)
    @test y ≈ link(d, x)
    @test inverse(b)(y) ≈ x
    @test logabsdetjac(b, x) ≈
        logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)
end

@testset "Multivariate" begin
    vector_dists = [
        Dirichlet(2, 3),
        Dirichlet([10.0, 0.1]),
        Dirichlet([0.1, 10.0]),
        MvNormal(randn(10), Diagonal(exp.(randn(10)))),
        MvLogNormal(MvNormal(randn(10), Diagonal(exp.(randn(10))))),
        MvTDist(1, randn(10), Matrix(Diagonal(exp.(randn(10))))),
        transformed(MvNormal(randn(10), Diagonal(exp.(randn(10))))),
        transformed(MvLogNormal(MvNormal(randn(10), Diagonal(exp.(randn(10)))))),
        transformed(reshape(product_distribution(fill(InverseGamma(2, 3), 6)), 2, 3)),
    ]

    for dist in vector_dists
        @testset "$dist: dist" begin
            td = transformed(dist)

            # single sample
            y = rand(td)
            x = inverse(td.transform)(y)
            @test y ≈ td.transform(x)
            @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

            # verify against AD
            # similar to what we do in test/transform.jl for Dirichlet
            if dist isa Dirichlet
                b = Bijectors.SimplexBijector()
                x = rand(dist)
                y = b(x)
                @test logabsdet(ForwardDiff.jacobian(b, x)[:, 1:(end - 1)])[1] ≈
                    logabsdetjac(b, x)
                @test logabsdet(ForwardDiff.jacobian(inverse(b), y)[1:(end - 1), :])[1] ≈
                    logabsdetjac(inverse(b), y)
            else
                b = bijector(dist)
                x = rand(dist)
                y = b(x)
                # `ForwardDiff.derivative` can lead to some numerical inaccuracy,
                # so we use a slightly higher `atol` than default.
                @test log(abs(det(ForwardDiff.jacobian(b, x)))) ≈ logabsdetjac(b, x) atol =
                    1e-6
                @test log(abs(det(ForwardDiff.jacobian(inverse(b), y)))) ≈
                    logabsdetjac(inverse(b), y) atol = 1e-6
            end
        end
    end
end

@testset "Matrix variate" begin
    v = 7.0
    S = Matrix(1.0I, 2, 2)
    S[1, 2] = S[2, 1] = 0.5

    matrix_dists = [
        Wishart(v, S),
        InverseWishart(v, S),
        LKJ(3, 1.0),
        reshape(MvNormal(zeros(6), I), 2, 3),
        product_distribution(fill(InverseGamma(2, 3), 6)),
    ]

    for dist in matrix_dists
        @testset "$dist: dist" begin
            td = transformed(dist)

            # single sample
            y = rand(td)
            x = inverse(td.transform)(y)
            @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

            # TODO: implement `logabsdetjac` for these
            # logpdf_with_jac
            # lp, logjac = logpdf_with_jac(td, y)
            # @test lp ≈ logpdf(td, y)
            # @test logjac ≈ logabsdetjacinv(td.transform, y)
        end
    end
end

@testset "ProductDistribution" begin
    d = product_distribution(fill(Dirichlet(ones(4)), 2, 3))
    x = rand(d)
    b = bijector(d)

    @test logpdf_with_trans(d, x, false) == logpdf(d, x)
    @test logpdf_with_trans(d, x, true) == logpdf(d, x) - logabsdetjac(b, x)
end
