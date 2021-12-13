using Test
using Random
using LinearAlgebra
using ForwardDiff
using ReverseDiff
using Tracker
using DistributionsAD

using Bijectors
using Bijectors: Log, Exp, Shift, Scale, Logit, SimplexBijector, PDBijector, Permute, PlanarLayer, RadialLayer, Stacked, TruncatedBijector, ADBijector, RationalQuadraticSpline, LeakyReLU

Random.seed!(123)

struct MyADBijector{AD, N, B <: Bijector{N}} <: ADBijector{AD, N}
    b::B
end
MyADBijector(d::Distribution) = MyADBijector{Bijectors.ADBackend()}(d)
MyADBijector{AD}(d::Distribution) where {AD} = MyADBijector{AD}(bijector(d))
MyADBijector{AD}(b::B) where {AD, N, B <: Bijector{N}} = MyADBijector{AD, N, B}(b)
(b::MyADBijector)(x) = b.b(x)
(b::Inverse{<:MyADBijector})(x) = inverse(b.orig.b)(x)

struct NonInvertibleBijector{AD} <: ADBijector{AD, 1} end

contains(predicate::Function, b::Bijector) = predicate(b)
contains(predicate::Function, b::Composed) = any(contains.(predicate, b.ts))
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

            # logpdf_with_jac
            lp, logjac = logpdf_with_jac(td, y)
            @test lp ≈ logpdf(td, y)
            @test logjac ≈ logabsdetjacinv(td.transform, y)

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

            # forward
            f = @inferred forward(td)
            @test f.x ≈ inverse(td.transform)(f.y)
            @test f.y ≈ td.transform(f.x)
            @test f.logabsdetjac ≈ logabsdetjac(td.transform, f.x)
            @test f.logpdf ≈ logpdf_with_trans(td.dist, f.x, true)
            @test f.logpdf ≈ logpdf(td.dist, f.x) - f.logabsdetjac

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

@testset "Batch computation" begin
    bs_xs = [
        (Scale(2.0), randn(3)),
        (Scale([1.0, 2.0]), randn(2, 3)),
        (Shift(2.0), randn(3)),
        (Shift([1.0, 2.0]), randn(2, 3)),
        (Log{0}(), exp.(randn(3))),
        (Log{1}(), exp.(randn(2, 3))),
        (Exp{0}(), randn(3)),
        (Exp{1}(), randn(2, 3)),
        (Log{1}() ∘ Exp{1}(), randn(2, 3)),
        (inverse(Logit(-1.0, 1.0)), randn(3)),
        (Identity{0}(), randn(3)),
        (Identity{1}(), randn(2, 3)),
        (PlanarLayer(2), randn(2, 3)),
        (RadialLayer(2), randn(2, 3)),
        (PlanarLayer(2) ∘ RadialLayer(2), randn(2, 3)),
        (Exp{1}() ∘ PlanarLayer(2) ∘ RadialLayer(2), randn(2, 3)),
        (SimplexBijector(), mapslices(z -> normalize(z, 1), rand(2, 3); dims = 1)),
        (stack(Exp{0}(), Scale(2.0)), randn(2, 3)),
        (Stacked((Exp{1}(), SimplexBijector()), (1:1, 2:3)),
         mapslices(z -> normalize(z, 1), rand(3, 2); dims = 1)),
        (RationalQuadraticSpline(randn(3), randn(3), randn(3 - 1), 2.), [-0.5, 0.5]),
        (LeakyReLU(0.1), randn(3)),
        (LeakyReLU(Float32(0.1)), randn(3)),
        (LeakyReLU(0.1; dim = Val(1)), randn(2, 3)),
    ]

    for (b, xs) in bs_xs
        @testset "$b" begin
            D = @inferred Bijectors.dimension(b)
            ib = @inferred inverse(b)

            @test Bijectors.dimension(ib) == D

            x = D == 0 ? xs[1] : xs[:, 1]

            y = @inferred b(x)
            ys = @inferred b(xs)
            @inferred(b(param(xs)))

            x_ = @inferred ib(y)
            xs_ = @inferred ib(ys)
            @inferred(ib(param(ys)))

            result = @inferred with_logabsdet_jacobian(b, x)
            results = @inferred with_logabsdet_jacobian(b, xs)

            iresult = @inferred with_logabsdet_jacobian(ib, y)
            iresults = @inferred with_logabsdet_jacobian(ib, ys)

            # Sizes
            @test size(y) == size(x)
            @test size(ys) == size(xs)

            @test size(x_) == size(x)
            @test size(xs_) == size(xs)

            @test size(result[1]) == size(x)
            @test size(results[1]) == size(xs)

            @test size(iresult[1]) == size(y)
            @test size(iresults[1]) == size(ys)

            # Values
            @test ys ≈ hcat([b(xs[:, i]) for i = 1:size(xs, 2)]...)
            @test ys ≈ results[1]

            if D == 0
                # Sizes
                @test y == ys[1]

                @test length(logabsdetjac(b, xs)) == length(xs)
                @test length(logabsdetjac(ib, ys)) == length(xs)

                @test @inferred(logabsdetjac(b, param(xs))) isa Union{Array, TrackedArray}
                @test @inferred(logabsdetjac(ib, param(ys))) isa Union{Array, TrackedArray}

                @test size(results[2]) == size(xs, )
                @test size(iresults[2]) == size(ys, )

                # Values
                b_logjac_ad = [(log ∘ abs)(ForwardDiff.derivative(b, xs[i])) for i = 1:length(xs)]
                ib_logjac_ad = [(log ∘ abs)(ForwardDiff.derivative(ib, ys[i])) for i = 1:length(ys)]
                @test logabsdetjac.(b, xs) == @inferred(logabsdetjac(b, xs))
                @test @inferred(logabsdetjac(b, xs)) ≈ b_logjac_ad atol=1e-9
                @test logabsdetjac.(ib, ys) == @inferred(logabsdetjac(ib, ys))
                @test @inferred(logabsdetjac(ib, ys)) ≈ ib_logjac_ad atol=1e-9

                @test logabsdetjac.(b, param(xs)) == @inferred(logabsdetjac(b, param(xs)))
                @test logabsdetjac.(ib, param(ys)) == @inferred(logabsdetjac(ib, param(ys)))

                @test results[2] ≈ vec(logabsdetjac.(b, xs))
                @test iresults[2] ≈ vec(logabsdetjac.(ib, ys))
            elseif D == 1
                @test y == ys[:, 1]
                # Comparing sizes instead of lengths ensures we catch errors s.t.
                # length(x) == 3 when size(x) == (1, 3).
                # Sizes
                @test size(logabsdetjac(b, xs)) == (size(xs, 2), )
                @test size(logabsdetjac(ib, ys)) == (size(xs, 2), )

                @test @inferred(logabsdetjac(b, param(xs))) isa Union{Array, TrackedArray}
                @test @inferred(logabsdetjac(ib, param(ys))) isa Union{Array, TrackedArray}

                @test size(results[2]) == (size(xs, 2), )
                @test size(iresults[2]) == (size(ys, 2), )

                # Test all values
                @test @inferred(logabsdetjac(b, xs)) ≈ vec(mapslices(z -> logabsdetjac(b, z), xs; dims = 1))
                @test @inferred(logabsdetjac(ib, ys)) ≈ vec(mapslices(z -> logabsdetjac(ib, z), ys; dims = 1))

                @test results[2] ≈ vec(mapslices(z -> logabsdetjac(b, z), xs; dims = 1))
                @test iresults[2] ≈ vec(mapslices(z -> logabsdetjac(ib, z), ys; dims = 1))

                # FIXME: `SimplexBijector` results in ∞ gradient if not in the domain
                if !contains(t -> t isa SimplexBijector, b)
                    b_logjac_ad = [logabsdet(ForwardDiff.jacobian(b, xs[:, i]))[1] for i = 1:size(xs, 2)]
                    @test logabsdetjac(b, xs) ≈ b_logjac_ad atol=1e-9

                    ib_logjac_ad = [logabsdet(ForwardDiff.jacobian(ib, ys[:, i]))[1] for i = 1:size(ys, 2)]
                    @test logabsdetjac(ib, ys) ≈ ib_logjac_ad atol=1e-9
                end
            else
                error("tests not implemented yet")
            end
        end
    end

    @testset "Composition" begin
        @test_throws DimensionMismatch (Exp{1}() ∘ Log{0}())

        # Check that type-stable composition stays type-stable
        cb1 = Composed((Exp(), Log())) ∘ Exp()
        @test cb1 isa Composed{<:Tuple}
        cb2 = Exp() ∘ Composed((Exp(), Log()))
        @test cb2 isa Composed{<:Tuple}
        cb3 = cb1 ∘ cb2
        @test cb3 isa Composed{<:Tuple}
        
        @test logabsdetjac(cb1, 1.) isa Real
        @test logabsdetjac(cb1, 1.) == 1.

        @test inverse(cb1) isa Composed{<:Tuple}
        @test inverse(cb2) isa Composed{<:Tuple}
        @test inverse(cb3) isa Composed{<:Tuple}

        # Check that type-unstable composition stays type-unstable
        cb1 = Composed([Exp(), Log()]) ∘ Exp()
        @test cb1 isa Composed{<:AbstractArray}
        cb2 = Exp() ∘ Composed([Exp(), Log()])
        @test cb2 isa Composed{<:AbstractArray}
        cb3 = cb1 ∘ cb2
        @test cb3 isa Composed{<:AbstractArray}
        
        @test logabsdetjac(cb1, 1.) isa Real
        @test logabsdetjac(cb1, 1.) == 1.

        @test inverse(cb1) isa Composed{<:AbstractArray}
        @test inverse(cb2) isa Composed{<:AbstractArray}
        @test inverse(cb3) isa Composed{<:AbstractArray}

        # combining the two
        @test_throws ErrorException (Log() ∘ Exp()) ∘ cb1
        @test_throws ErrorException cb1 ∘ (Log() ∘ Exp())
    end

    @testset "Batch-computation with Tracker.jl" begin
        @testset "Scale" begin
            # 0-dim with `Real` parameter
            b = Scale(param(2.0))
            lj = logabsdetjac(b, 1.0)
            Tracker.back!(lj, 1.0)
            @test Tracker.extract_grad!(b.a) == 0.5

            # 0-dim with `Real` parameter for batch-computation
            lj = logabsdetjac(b, [1.0, 2.0, 3.0])
            Tracker.back!(lj, [1.0, 1.0, 1.0])
            @test Tracker.extract_grad!(b.a) == sum([0.5, 0.5, 0.5])


            # 1-dim with `Vector` parameter
            x = [3.0, 4.0, 5.0]
            xs = [3.0 4.0; 4.0 7.0; 5.0 8.0]
            a = [2.0, 3.0, 5.0]

            b = Scale(param(a))
            lj = logabsdetjac(b, x)
            Tracker.back!(lj)
            @test Tracker.extract_grad!(b.a) == ForwardDiff.gradient(a -> logabsdetjac(Scale(a), x), a)

            # batch
            lj = logabsdetjac(b, xs)
            Tracker.back!(mean(lj), 1.0)
            @test Tracker.extract_grad!(b.a) == ForwardDiff.gradient(a -> mean(logabsdetjac(Scale(a), xs)), a)

            # Forward when doing a composition
            y, logjac = logabsdetjac(b, xs)
            Tracker.back!(mean(logjac), 1.0)
            @test Tracker.extract_grad!(b.a) == ForwardDiff.gradient(a -> mean(logabsdetjac(Scale(a), xs)), a)
        end

        @testset "Shift" begin
            b = Shift(param(1.0))
            lj = logabsdetjac(b, 1.0)
            Tracker.back!(lj, 1.0)
            @test Tracker.extract_grad!(b.a) == 0.0

            # 0-dim with `Real` parameter for batch-computation
            lj = logabsdetjac(b, [1.0, 2.0, 3.0])
            @test lj isa TrackedArray
            Tracker.back!(lj, [1.0, 1.0, 1.0])
            @test Tracker.extract_grad!(b.a) == 0.0

            # 1-dim with `Vector` parameter
            b = Shift(param([2.0, 3.0, 5.0]))
            lj = logabsdetjac(b, [3.0, 4.0, 5.0])
            Tracker.back!(lj)
            @test Tracker.extract_grad!(b.a) == zeros(3)

            lj = logabsdetjac(b, [3.0 4.0 5.0; 6.0 7.0 8.0])
            @test lj isa TrackedArray
            Tracker.back!(lj, [1.0, 1.0, 1.0])
            @test Tracker.extract_grad!(b.a) == zeros(3)
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

            # logpdf_with_jac
            lp, logjac = logpdf_with_jac(td, y)
            @test lp ≈ logpdf(td, y)
            @test logjac ≈ logabsdetjacinv(td.transform, y)

            # multi-sample
            y = rand(td, 10)
            x = inverse(td.transform)(y)
            @test inverse(td.transform)(param(y)) isa TrackedArray
            @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

            # forward
            f = forward(td)
            @test f.x ≈ inverse(td.transform)(f.y)
            @test f.y ≈ td.transform(f.x)
            @test f.logabsdetjac ≈ logabsdetjac(td.transform, f.x)
            @test f.logpdf ≈ logpdf_with_trans(td.dist, f.x, true)

            # verify against AD
            # similar to what we do in test/transform.jl for Dirichlet
            if dist isa Dirichlet
                b = Bijectors.SimplexBijector{1, false}()
                x = rand(dist)
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

            # multi-sample
            y = rand(td, 10)
            x = inverse(td.transform)(y)
            @test inverse(td.transform)(param.(y)) isa Vector{<:TrackedArray}
            @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)
        end
    end
end

@testset "Composition <: Bijector" begin
    d = Beta()
    td = transformed(d)

    x = rand(d)
    y = td.transform(x)

    b = @inferred Bijectors.composel(td.transform, Bijectors.Identity{0}())
    ib = @inferred inverse(b)

    @test with_logabsdet_jacobian(b, x) == with_logabsdet_jacobian(td.transform, x)
    @test with_logabsdet_jacobian(ib, y) == with_logabsdet_jacobian(inverse(td.transform), y)

    @test with_logabsdet_jacobian(b, x) == with_logabsdet_jacobian(Bijectors.composer(b.ts...), x)

    # inverse works fine for composition
    cb = @inferred b ∘ ib
    @test cb(x) ≈ x

    cb2 = @inferred cb ∘ cb
    @test cb(x) ≈ x

    # ensures that the `logabsdetjac` is correct
    x = rand(d)
    b = inverse(bijector(d))
    @test logabsdetjac(b ∘ b, x) ≈ logabsdetjac(b, b(x)) + logabsdetjac(b, x)

    # order of composed evaluation
    b1 = MyADBijector(d)
    b2 = MyADBijector(Gamma())

    cb = inverse(b1) ∘ b2
    @test cb(x) ≈ inverse(b1)(b2(x))

    # contrived example
    b = bijector(d)
    cb = @inferred inverse(b) ∘ b
    cb = @inferred cb ∘ cb
    @test @inferred(cb ∘ cb ∘ cb ∘ cb ∘ cb)(x) ≈ x

    # forward for tuple and array
    d = Beta()
    b = @inferred inverse(bijector(d))
    b⁻¹ = @inferred inverse(b)
    x = rand(d)

    cb_t = b⁻¹ ∘ b⁻¹
    f_t = with_logabsdet_jacobian(cb_t, x)

    cb_a = Composed([b⁻¹, b⁻¹])
    f_a = with_logabsdet_jacobian(cb_a, x)

    @test f_t == f_a

    # `composer` and `composel`
    cb_l = Bijectors.composel(b⁻¹, b⁻¹, b)
    cb_r = Bijectors.composer(reverse(cb_l.ts)...)
    y = cb_l(x)
    @test y == Bijectors.composel(cb_r.ts...)(x)

    k = length(cb_l.ts)
    @test all([cb_l.ts[i] == cb_r.ts[i] for i = 1:k])
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
    sb = @inferred stack(Bijectors.Exp(), Bijectors.Log(), Bijectors.Shift(5.0))
    res = with_logabsdet_jacobian(sb, x)
    @test sb(param(x)) isa TrackedArray
    @test sb(x) == [exp(x[1]), log(x[2]), x[3] + 5.0]
    @test res[1] == [exp(x[1]), log(x[2]), x[3] + 5.0]
    @test logabsdetjac(sb, x) == sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges[i]])) for i = 1:3])
    @test res[2] == logabsdetjac(sb, x)


    # TODO: change when we have dimensionality in the type
    sb = @inferred Stacked((Bijectors.Exp(), Bijectors.SimplexBijector()), (1:1, 2:3))
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
    sb = Stacked([Bijectors.Exp(), Bijectors.SimplexBijector()], [1:1, 2:3])
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
    sb = Stacked([Bijectors.Exp(), Bijectors.SimplexBijector()], (1:1, 2:3))
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
    sb = Stacked((Bijectors.Exp(), Bijectors.SimplexBijector()), [1:1, 2:3])
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
        @test_throws MethodError stack(ibs...)
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
        Identity{0}(),
        Identity{1}(),
        Identity{2}(),
        Exp{0}(),
        Exp{1}(),
        Exp{2}(),
        Log{0}(),
        Log{1}(),
        Log{2}(),
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
        Stacked((Exp{0}(), Log{0}())),
        Stacked((Log{0}(), Exp{0}())),
        Stacked([Exp{0}(), Log{0}()]),
        Stacked([Log{0}(), Exp{0}()]),
        Composed((Exp{0}(), Log{0}())),
        Composed((Log{0}(), Exp{0}())),
        # Composed([Exp{0}(), Log{0}()]),
        # Composed([Log{0}(), Exp{0}()]),
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
    b = Bijectors.Scale{Float64,0}(4.2)
    x = 0.3

    test_inverse(b, x)
    test_with_logabsdet_jacobian(b, x, (f::Bijectors.Scale, x) -> f.a)
end


@testset "deprecations" begin
    b = Bijectors.Exp()
    x = 0.3

    @test @test_deprecated(forward(b, x)) == NamedTuple{(:rv, :logabsdetjac)}(with_logabsdet_jacobian(b, x))
    @test @test_deprecated(inv(b)) == inverse(b)
end
