using Bijectors: OrderedBijector, ordered
using LinearAlgebra

using LogDensityProblems: LogDensityProblems
using AbstractMCMC: AbstractMCMC
using AdvancedHMC: AdvancedHMC

using MCMCDiagnosticTools: MCMCDiagnosticTools

struct OrderedTestProblem{D}
    d::D
end

# LogDensityProblems.jl interface.
function LogDensityProblems.capabilities(p::OrderedTestProblem)
    return LogDensityProblems.LogDensityOrder{0}()
end
LogDensityProblems.dimension(p::OrderedTestProblem) = length(p.d)
function LogDensityProblems.logdensity(p::OrderedTestProblem, θ)
    return logpdf(transformed(ordered(p.d)), θ)
end

to_constrained(p::OrderedTestProblem, θ) = inverse(bijector(ordered(p.d)))(θ)

@testset "OrderedBijector" begin
    b = OrderedBijector()

    # Length 1
    x = randn(1)
    test_bijector(b, x; test_not_identity=false)

    # Larger
    x = randn(5)
    test_bijector(b, x)

    y = b(x)
    @test sort(y) == y
end

@testset "ordered" begin
    @testset "$(typeof(d))" for d in [
        MvNormal(1:5, Diagonal(6:10)),
        MvTDist(1, collect(1.0:5), Matrix(I(5))),
        product_distribution(fill(Normal(), 5)),
        product_distribution(fill(TDist(1), 5)),
        # positive supports
        product_distribution(fill(LogNormal(), 5)),
        product_distribution(fill(InverseGamma(2, 3), 5)),
        # negative supports
        product_distribution(fill(-1 * InverseGamma(2, 3), 5)),
        # bounded supports
        product_distribution(fill(Uniform(1, 2), 5)),
        product_distribution(fill(Beta(), 5)),
        # different transformations
        product_distribution(fill(transformed(InverseGamma(2, 3), Bijectors.Scale(3)), 5)),
        product_distribution(fill(transformed(InverseGamma(2, 3), Bijectors.Scale(-3)), 5)),
        product_distribution(fill(transformed(InverseGamma(2, 3), Bijectors.Shift(3)), 5)),
        product_distribution(fill(transformed(InverseGamma(2, 3), Bijectors.Shift(-3)), 5)),
    ]
        d_ordered = ordered(d)
        @test d_ordered isa Bijectors.OrderedDistribution
        @test d_ordered.dist === d
        num_tries = 100
        for _ in 1:num_tries
            y = randn(size(d))
            x = inverse(bijector(d_ordered))(y)
            @test issorted(x)
            @test isfinite(logpdf(d_ordered, x))
        end
        # Check that `logpdf` correctly identifies out-of-support values.
        @test !isfinite(logpdf(d_ordered, sort(rand(d); rev=true)))
    end

    @testset "non-identity bijector is not supported" begin
        d = Dirichlet(ones(5))
        @test_throws ArgumentError ordered(d)
    end

    @testset "correctness" begin
        num_samples = 10_000
        num_adapts = 1_000
        @testset "k = $k" for k in [2, 3, 5]
            @testset "$(typeof(dist))" for dist in [
                # Unconstrained
                MvNormal(1:k, Diagonal(1:k)),
                MvNormal(1:k),
                # positive support
                product_distribution(fill(Exponential(), k)),
                # bounded
                product_distribution(fill(Beta(), k)),
                # using `Scale`
                product_distribution(
                    fill(transformed(Exponential(), Bijectors.Scale(3)), k)
                ),
                # negative support
                product_distribution(fill(-1 * Exponential(), k)),
            ]
                prob = OrderedTestProblem(dist)
                sampler = AdvancedHMC.NUTS(0.8)
                initial_params = rand(ordered(dist))
                transitions = sample(
                    prob,
                    sampler,
                    num_samples;
                    initial_params=initial_params,
                    discard_initial=num_adapts,
                    n_adapts=num_adapts,
                    progress=false,
                )
                xs = mapreduce(hcat, transitions) do t
                    to_constrained(prob, t.z.θ)
                end
                @test all(issorted, eachcol(xs))

                # `rand` uses rejection sampling => exact.
                dist_ordered = ordered(dist)
                xs_true = rand(dist_ordered, num_samples)
                @test all(issorted, eachcol(xs_true))

                # Compare MCMC quantiles to exact quantiles.
                qts = [0.05, 0.25, 0.5, 0.75, 0.95]
                qs_true = mapslices(Base.Fix2(quantile, qts), xs_true; dims=2)
                qs = mapslices(Base.Fix2(quantile, qts), xs; dims=2)
                qs_mcse = mapslices(xs; dims=2) do x
                    map(qts) do q
                        MCMCDiagnosticTools.mcse(x; kind=Base.Fix2(quantile, q))
                    end
                end
                # Check that the quantiles are reasonable, i.e. within
                # 5 standard errors of the true quantiles (and that the MCSE is
                # not too large).
                @info "Checking quantiles" qs_true qs qs_mcse
                for i in 1:k
                    for j in 1:length(qts)
                        @test qs_mcse[i, j] < abs(qs_true[i, end] - qs_true[i, 1]) / 2
                        @test abs(qs[i, j] - qs_true[i, j]) < 5 * qs_mcse[i, j]
                    end
                end
            end
        end
    end
end
