# using Pkg; Pkg.activate("..")
# using TestEnv; TestEnv.activate()

using Test
using Random
using LinearAlgebra
using ForwardDiff
using ReverseDiff
using Tracker
using DistributionsAD

using Bijectors
using Bijectors: Shift, Scale, Logit, SimplexBijector, PDBijector, Permute, PlanarLayer, RadialLayer, Stacked, TruncatedBijector, ADBijector, RationalQuadraticSpline, LeakyReLU

Random.seed!(123)

struct MyADBijector{AD,B} <: ADBijector{AD}
    b::B
end
MyADBijector(d::Distribution) = MyADBijector{Bijectors.ADBackend()}(d)
MyADBijector{AD}(d::Distribution) where {AD} = MyADBijector{AD}(bijector(d))
MyADBijector{AD}(b) where {AD} = MyADBijector{AD, typeof(b)}(b)
(b::MyADBijector)(x) = b.b(x)
Bijectors.transform(b::MyADBijector, x) = b.b(x)
Bijectors.transform(b::Inverse{<:MyADBijector}, x) = inverse(b.orig.b)(x)

struct NonInvertibleBijector{AD} <: ADBijector{AD} end

contains(predicate::Function, b::Bijector) = predicate(b)
contains(predicate::Function, b::ComposedFunction) = any(contains.(predicate, b.ts))
contains(predicate::Function, b::Stacked) = any(contains.(predicate, b.bs))

# Scalar tests
@testset "<: ADBijector{AD}" begin
    (b::NonInvertibleBijector)(x) = clamp.(x, 0, 1)

    b = NonInvertibleBijector{Bijectors.ADBackend()}()
    @test_throws Bijectors.SingularJacobianException logabsdetjac(b, [1.0, 10.0])
end

@testset "Univariate" begin
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
        truncated(Normal(0, 1), -Inf, 2),
        transformed(Beta(2,2)),
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
            @test logpdf(d, inverse(b)(y)) + logabsdetjacinv(b, y) ≈ logpdf_with_trans(d, x, true)
            @test logpdf(d, x) - logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, true)

            # verify against AD
            d = dist
            b = bijector(d)
            x = rand(d)
            y = b(x)
            # `ForwardDiff.derivative` can lead to some numerical inaccuracy,
            # so we use a slightly higher `atol` than default.
            @test log(abs(ForwardDiff.derivative(b, x))) ≈ logabsdetjac(b, x) atol=1e-6
            @test log(abs(ForwardDiff.derivative(inverse(b), y))) ≈ logabsdetjac(inverse(b), y) atol=1e-6
        end

        @testset "$dist: ForwardDiff AD" begin
            x = rand(dist)
            b = MyADBijector{Bijectors.ADBackend(:forwarddiff)}(dist)
            
            @test abs(det(Bijectors.jacobian(b, x))) > 0
            @test logabsdetjac(b, x) ≠ Inf

            y = b(x)
            b⁻¹ = inverse(b)
            @test abs(det(Bijectors.jacobian(b⁻¹, y))) > 0
            @test logabsdetjac(b⁻¹, y) ≠ Inf
        end

        @testset "$dist: Tracker AD" begin
            x = rand(dist)
            b = MyADBijector{Bijectors.ADBackend(:reversediff)}(dist)
            
            @test abs(det(Bijectors.jacobian(b, x))) > 0
            @test logabsdetjac(b, x) ≠ Inf

            y = b(x)
            b⁻¹ = inverse(b)
            @test abs(det(Bijectors.jacobian(b⁻¹, y))) > 0
            @test logabsdetjac(b⁻¹, y) ≠ Inf
        end
    end
end

@testset "Truncated" begin
    d = truncated(Normal(), -1, 1)
    b = bijector(d)
    x = rand(d)
    y = b(x)
    @test y ≈ link(d, x)
    @test inverse(b)(y) ≈ x
    @test logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)

    d = truncated(Normal(), -Inf, 1)
    b = bijector(d)
    x = rand(d)
    y = b(x)
    @test y ≈ link(d, x)
    @test inverse(b)(y) ≈ x
    @test logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)

    d = truncated(Normal(), 1, Inf)
    b = bijector(d)
    x = rand(d)
    y = b(x)
    @test y ≈ link(d, x)
    @test inverse(b)(y) ≈ x
    @test logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)
end

@testset "Multivariate" begin
    vector_dists = [
        Dirichlet(2, 3),
        Dirichlet([1000 * one(Float64), eps(Float64)]),
        Dirichlet([eps(Float64), 1000 * one(Float64)]),
        MvNormal(randn(10), Diagonal(exp.(randn(10)))),
        MvLogNormal(MvNormal(randn(10), Diagonal(exp.(randn(10))))),
        Dirichlet([1000 * one(Float64), eps(Float64)]), 
        Dirichlet([eps(Float64), 1000 * one(Float64)]),
        transformed(MvNormal(randn(10), Diagonal(exp.(randn(10))))),
        transformed(MvLogNormal(MvNormal(randn(10), Diagonal(exp.(randn(10))))))
    ]

    for dist in vector_dists
        @testset "$dist: dist" begin
            td = transformed(dist)

            # single sample
            y = rand(td)
            x = inverse(td.transform)(y)
            @test inverse(td.transform)(param(y)) isa TrackedArray
            @test y ≈ td.transform(x)
            @test td.transform(param(x)) isa TrackedArray
            @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

            # verify against AD
            # similar to what we do in test/transform.jl for Dirichlet
            if dist isa Dirichlet
                b = Bijectors.SimplexBijector{false}()
                # HACK(torfjelde): Calling `rand(dist)` will sometimes lead to `[0.999..., 0.0]`
                # which in turn will lead to differences between `ForwardDiff.jacobian`
                # and `logabsdetjac` due to how we handle the boundary values in `SimplexBijector`.
                # We therefore test the realizations _on_ the boundary rather if we're near the boundary.
                x = any(rand(dist) .> 0.9999) ? [0.0, 1.0][sortperm(rand(dist))] : rand(dist)
                y = b(x)
                @test b(param(x)) isa TrackedArray
                @test log(abs(det(ForwardDiff.jacobian(b, x)))) ≈ logabsdetjac(b, x)
                @test log(abs(det(ForwardDiff.jacobian(inverse(b), y)))) ≈ logabsdetjac(inverse(b), y)
            else
                b = bijector(dist)
                x = rand(dist)
                y = b(x)
                # `ForwardDiff.derivative` can lead to some numerical inaccuracy,
                # so we use a slightly higher `atol` than default.
                @test b(param(x)) isa TrackedArray
                @test log(abs(det(ForwardDiff.jacobian(b, x)))) ≈ logabsdetjac(b, x) atol=1e-6
                @test log(abs(det(ForwardDiff.jacobian(inverse(b), y)))) ≈ logabsdetjac(inverse(b), y) atol=1e-6
            end
        end
    end
end

@testset "Matrix variate" begin
    v = 7.0
    S = Matrix(1.0I, 2, 2)
    S[1, 2] = S[2, 1] = 0.5

    matrix_dists = [
        Wishart(v,S),
        InverseWishart(v,S),
        TuringWishart(v,S),
        TuringInverseWishart(v,S),
        LKJ(3, 1.)
    ]

    for dist in matrix_dists
        @testset "$dist: dist" begin
            td = transformed(dist)

            # single sample
            y = rand(td)
            x = inverse(td.transform)(y)
            @test inverse(td.transform)(param(y)) isa TrackedArray
            @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

            # TODO: implement `logabsdetjac` for these
            # logpdf_with_jac
            # lp, logjac = logpdf_with_jac(td, y)
            # @test lp ≈ logpdf(td, y)
            # @test logjac ≈ logabsdetjacinv(td.transform, y)
        end
    end
end

@testset "Stacked <: Bijector" begin
    # `logabsdetjac` withOUT AD
    d = Beta()
    b = bijector(d)
    x = rand(d)
    y = b(x)

    sb1 = @inferred stack(b, b, inverse(b), inverse(b))             # <= Tuple
    res1 = with_logabsdet_jacobian(sb1, [x, x, y, y])
    @test sb1(param([x, x, y, y])) isa TrackedArray

    @test sb1([x, x, y, y]) ≈ res1[1]
    @test logabsdetjac(sb1, [x, x, y, y]) ≈ 0 atol=1e-6
    @test res1[2] ≈ 0 atol=1e-6

    sb2 = Stacked([b, b, inverse(b), inverse(b)])        # <= Array
    res2 = with_logabsdet_jacobian(sb2, [x, x, y, y])
    @test sb2(param([x, x, y, y])) isa TrackedArray

    @test sb2([x, x, y, y]) ≈ res2[1]
    @test logabsdetjac(sb2, [x, x, y, y]) ≈ 0.0 atol=1e-12
    @test res2[2] ≈ 0.0 atol=1e-12

    # `logabsdetjac` with AD
    b = MyADBijector(d)
    y = b(x)
    
    sb1 = stack(b, b, inverse(b), inverse(b))             # <= Tuple
    res1 = with_logabsdet_jacobian(sb1, [x, x, y, y])
    @test sb1(param([x, x, y, y])) isa TrackedArray

    @test sb1([x, x, y, y]) == res1[1]
    @test logabsdetjac(sb1, [x, x, y, y]) ≈ 0 atol=1e-12
    @test res1[2] ≈ 0.0 atol=1e-12

    sb2 = Stacked([b, b, inverse(b), inverse(b)])        # <= Array
    res2 = with_logabsdet_jacobian(sb2, [x, x, y, y])
    @test sb2(param([x, x, y, y])) isa TrackedArray

    @test sb2([x, x, y, y]) == res2[1]
    @test logabsdetjac(sb2, [x, x, y, y]) ≈ 0.0 atol=1e-12
    @test res2[2] ≈ 0.0 atol=1e-12

    # value-test
    x = ones(3)
    sb = @inferred stack(elementwise(exp), elementwise(log), Shift(5.0))
    res = with_logabsdet_jacobian(sb, x)
    @test sb(param(x)) isa TrackedArray
    @test sb(x) == [exp(x[1]), log(x[2]), x[3] + 5.0]
    @test res[1] == [exp(x[1]), log(x[2]), x[3] + 5.0]
    @test logabsdetjac(sb, x) == sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:3])
    @test res[2] == logabsdetjac(sb, x)


    # TODO: change when we have dimensionality in the type
    sb = @inferred Stacked((elementwise(exp), SimplexBijector()), (1:1, 2:3))
    x = ones(3) ./ 3.0
    res = @inferred with_logabsdet_jacobian(sb, x)
    @test sb(param(x)) isa TrackedArray
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) == sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws AssertionError sb(x)

    # Array-version
    sb = Stacked([elementwise(exp), SimplexBijector()], [1:1, 2:3])
    x = ones(3) ./ 3.0
    res = with_logabsdet_jacobian(sb, x)
    @test sb(param(x)) isa TrackedArray
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) == sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws AssertionError sb(x)

    # Mixed versions
    # Tuple, Array
    sb = Stacked([elementwise(exp), SimplexBijector()], (1:1, 2:3))
    x = ones(3) ./ 3.0
    res = with_logabsdet_jacobian(sb, x)
    @test sb(param(x)) isa TrackedArray
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) == sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws AssertionError sb(x)

    # Array, Tuple
    sb = Stacked((elementwise(exp), SimplexBijector()), [1:1, 2:3])
    x = ones(3) ./ 3.0
    res = with_logabsdet_jacobian(sb, x)
    @test sb(param(x)) isa TrackedArray
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) == sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws AssertionError sb(x)


    @testset "Stacked: ADVI with MvNormal" begin
        # MvNormal test
        dists = [
            Beta(),
            Beta(),
            Beta(),
            InverseGamma(),
            InverseGamma(),
            Gamma(),
            Gamma(),
            InverseGamma(),
            Cauchy(),
            Gamma(),
            MvNormal(zeros(2), I)
        ]

        ranges = []
        idx = 1
        for i = 1:length(dists)
            d = dists[i]
            push!(ranges, idx:idx + length(d) - 1)
            idx += length(d)
        end
        ranges = tuple(ranges...)

        num_params = ranges[end][end]
        d = MvNormal(zeros(num_params), I)

        # Stacked{<:Array}
        bs = bijector.(dists)     # constrained-to-unconstrained bijectors for dists
        ibs = inverse.(bs)            # invert, so we get unconstrained-to-constrained
        sb = Stacked(ibs, ranges) # => Stacked <: Bijector
        x = rand(d)

        @test sb isa Stacked

        td = transformed(d, sb)  # => MultivariateTransformed <: Distribution{Multivariate, Continuous}
        @test td isa Distribution{Multivariate, Continuous}

        # check that wrong ranges fails
        sb = Stacked(ibs)
        x = rand(d)
        @test_throws AssertionError sb(x)

        # Stacked{<:Tuple}
        bs = bijector.(tuple(dists...))
        ibs = inverse.(bs)
        sb = @inferred Stacked(ibs, ranges)
        isb = @inferred inverse(sb)
        @test sb isa Stacked{<:Tuple}

        # inverse
        td = @inferred transformed(d, sb)
        y = @inferred rand(td)
        x = @inferred isb(y)
        @test sb(x) ≈ y

        # verification of computation
        x = rand(d)
        y = sb(x)
        y_ = vcat([ibs[i](x[ranges[i]]) for i = 1:length(dists)]...)
        x_ = vcat([bs[i](y[ranges[i]]) for i = 1:length(dists)]...)
        @test x ≈ x_
        @test y ≈ y_

        # AD verification
        @test log(abs(det(ForwardDiff.jacobian(sb, x)))) ≈ logabsdetjac(sb, x)
        @test log(abs(det(ForwardDiff.jacobian(isb, y)))) ≈ logabsdetjac(isb, y)

        # Ensure `Stacked` works for a single bijector
        d = (MvNormal(zeros(2), I),)
        sb = Stacked(bijector.(d), (1:2, ))
        x = [.5, 1.]
        @test sb(x) == x
        @test logabsdetjac(sb, x) == 0
        @test with_logabsdet_jacobian(sb, x) == (x, zero(eltype(x)))
    end
end

@testset "Example: ADVI single" begin
    # Usage in ADVI
    d = Beta()
    b = bijector(d)                # [0, 1] → ℝ
    ib = inverse(b)                    # ℝ → [0, 1]
    td = transformed(Normal(), ib) # x ∼ 𝓝(0, 1) then f(x) ∈ [0, 1]
    x = rand(td)                   # ∈ [0, 1]
    @test 0 ≤ x ≤ 1
end

@testset "Jacobians of SimplexBijector" begin
    b = SimplexBijector()
    ib = inverse(b)

    x = ib(randn(10))
    y = b(x)

    @test Bijectors.jacobian(b, x) ≈ ForwardDiff.jacobian(b, x)
    @test Bijectors.jacobian(ib, y) ≈ ForwardDiff.jacobian(ib, y)

    # Just some additional computation so we also ensure the pullbacks are the same
    weights = randn(10)

    # Tracker.jl
    x_tracked = Tracker.param(x)
    z = sum(weights .* b(x_tracked))
    Tracker.back!(z)
    Δ_tracker = Tracker.grad(x_tracked)

    # ForwardDiff.jl
    Δ_forwarddiff = ForwardDiff.gradient(z -> sum(weights .* b(z)), x)

    # Compare
    @test Δ_forwarddiff ≈ Δ_tracker

    # Tracker.jl
    y_tracked = Tracker.param(y)
    z = sum(weights .* ib(y_tracked))
    Tracker.back!(z)
    Δ_tracker = Tracker.grad(y_tracked)

    # ForwardDiff.jl
    Δ_forwarddiff = ForwardDiff.gradient(z -> sum(weights .* ib(z)), y)

    @test Δ_forwarddiff ≈ Δ_tracker
end

@testset "Equality" begin
    bs = [
        Identity(),
        elementwise(exp),
        elementwise(log),
        Scale(2.0),
        Scale(3.0),
        Scale(rand(2,2)),
        Scale(rand(2,2)),
        Shift(2.0),
        Shift(3.0),
        Shift(rand(2)),
        Shift(rand(2)),
        Logit(1.0, 2.0),
        Logit(1.0, 3.0),
        Logit(2.0, 3.0),
        Logit(0.0, 2.0),
        InvertibleBatchNorm(2),
        InvertibleBatchNorm(3),
        PDBijector(),
        Permute([1.0, 2.0, 3.0]),
        Permute([2.0, 3.0, 4.0]),
        PlanarLayer(2),
        PlanarLayer(3),
        RadialLayer(2),
        RadialLayer(3),
        SimplexBijector(),
        Stacked((elementwise(exp), elementwise(log))),
        Stacked((elementwise(log), elementwise(exp))),
        Stacked([elementwise(exp), elementwise(log)]),
        Stacked([elementwise(log), elementwise(exp)]),
        elementwise(exp) ∘ elementwise(log),
        elementwise(log) ∘ elementwise(exp),
        TruncatedBijector(1.0, 2.0),
        TruncatedBijector(1.0, 3.0),
        TruncatedBijector(0.0, 2.0),
    ]
    for i in 1:length(bs), j in 1:length(bs)
        if i == j
            @test bs[i] == deepcopy(bs[j])
            @test inverse(bs[i]) == inverse(deepcopy(bs[j]))
        else
            @test bs[i] != bs[j]
        end
    end
end

@testset "test_inverse and test_with_logabsdet_jacobian" begin
    b = Bijectors.Scale{Float64,}(4.2)
    x = 0.3

    InverseFunctions.test_inverse(b, x)
    ChangesOfVariables.test_with_logabsdet_jacobian(b, x, (f::Bijectors.Scale, x) -> f.a)
end
