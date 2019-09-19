using Test
using Bijectors
using Random
using LinearAlgebra
using ForwardDiff

using Bijectors: Log, Exp, Shift, Scale, Logit, SimplexBijector

Random.seed!(123)

struct NonInvertibleBijector{AD} <: ADBijector{AD, 1} end

# Scalar tests
@testset "Interface" begin
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
            TruncatedNormal(0, 1, -Inf, 2),
        ]
        
        for dist in uni_dists
            @testset "$dist: dist" begin
                td = transformed(dist)

                # single sample
                y = rand(td)
                x = inv(td.transform)(y)
                @test y == td.transform(x)
                @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

                # logpdf_with_jac
                lp, logjac = logpdf_with_jac(td, y)
                @test lp ≈ logpdf(td, y)
                @test logjac ≈ logabsdetjacinv(td.transform, y)

                # multi-sample
                y = rand(td, 10)
                x = inv(td.transform).(y)
                @test logpdf.(td, y) ≈ logpdf_with_trans.(dist, x, true)

                # logpdf corresponds to logpdf_with_trans
                d = dist
                b = bijector(d)
                x = rand(d)
                y = b(x)
                @test logpdf(d, inv(b)(y)) + logabsdetjacinv(b, y) ≈ logpdf_with_trans(d, x, true)
                @test logpdf(d, x) - logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, true)

                # forward
                f = forward(td)
                @test f.x ≈ inv(td.transform)(f.y)
                @test f.y ≈ td.transform(f.x)
                @test f.logabsdetjac ≈ logabsdetjac(td.transform, f.x)
                @test f.logpdf ≈ logpdf_with_trans(td.dist, f.x, true)
                @test f.logpdf ≈ logpdf(td.dist, f.x) - f.logabsdetjac

                # verify against AD
                d = dist
                b = bijector(d)
                x = rand(d)
                y = b(x)
                @test log(abs(ForwardDiff.derivative(b, x))) ≈ logabsdetjac(b, x)
                @test log(abs(ForwardDiff.derivative(inv(b), y))) ≈ logabsdetjac(inv(b), y)
            end

            @testset "$dist: ForwardDiff AD" begin
                x = rand(dist)
                b = DistributionBijector{Bijectors.ADBackend(:forward_diff), typeof(dist), length(size(dist))}(dist)
                
                @test abs(det(Bijectors.jacobian(b, x))) > 0
                @test logabsdetjac(b, x) ≠ Inf

                y = b(x)
                b⁻¹ = inv(b)
                @test abs(det(Bijectors.jacobian(b⁻¹, y))) > 0
                @test logabsdetjac(b⁻¹, y) ≠ Inf
            end

            @testset "$dist: Tracker AD" begin
                x = rand(dist)
                b = DistributionBijector{Bijectors.ADBackend(:reverse_diff), typeof(dist), length(size(dist))}(dist)
                
                @test abs(det(Bijectors.jacobian(b, x))) > 0
                @test logabsdetjac(b, x) ≠ Inf

                y = b(x)
                b⁻¹ = inv(b)
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
            (inv(Logit(-1.0, 1.0)), randn(3)),
            (Identity{0}(), randn(3)),
            (Identity{1}(), randn(2, 3)),
            (PlanarLayer(2), randn(2, 3)),
            (RadialLayer(2), randn(2, 3)),
            (PlanarLayer(2) ∘ RadialLayer(2), randn(2, 3)),
            (Exp{1}() ∘ PlanarLayer(2) ∘ RadialLayer(2), randn(2, 3)),
            (SimplexBijector(), mapslices(z -> normalize(z, 1), rand(2, 3); dims = 1))
        ]

        for (b, xs) in bs_xs
            @testset "$b" begin
                D = Bijectors.dimension(b)
                ib = inv(b)

                @test Bijectors.dimension(ib) == D

                x = D == 0 ? xs[1] : xs[:, 1]

                y = b(x)
                ys = b(xs)

                x_ = ib(y)
                xs_ = ib(ys)

                @test size(y) == size(x)
                @test size(ys) == size(xs)
                @test size(x_) == size(x)
                @test size(xs_) == size(xs)

                if D == 0
                    @test y == ys[1]

                    @test length(logabsdetjac(b, xs)) == length(xs)
                    @test logabsdetjac(b, x) == logabsdetjac(b, xs)[1]

                    @test length(logabsdetjac(ib, ys)) == length(xs)
                    @test logabsdetjac(ib, y) == logabsdetjac(ib, ys)[1]
                elseif Bijectors.dimension(b) == 1
                    @test y == ys[:, 1]
                    # Comparing sizes instead of lengths ensures we catch errors s.t.
                    # length(x) == 3 when size(x) == (1, 3).
                    # We want the return value to
                    @test size(logabsdetjac(b, xs)) == (size(xs, 2), )
                    @test logabsdetjac(b, x) == logabsdetjac(b, xs)[1]

                    @test size(logabsdetjac(ib, ys)) == (size(xs, 2), )
                    @test logabsdetjac(ib, y) == logabsdetjac(ib, ys)[1]
                else
                    error("tests not implemented yet")
                end
            end
        end

        @testset "Composition" begin
            @test_throws DimensionMismatch (Exp{1}() ∘ Log{0}())
        end
    end

    @testset "Truncated" begin
        d = Truncated(Normal(), -1, 1)
        b = bijector(d)
        x = rand(d)
        y = b(x)
        @test y ≈ link(d, x)
        @test inv(b)(y) ≈ x
        @test logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)

        d = Truncated(Normal(), -Inf, 1)
        b = bijector(d)
        x = rand(d)
        y = b(x)
        @test y ≈ link(d, x)
        @test inv(b)(y) ≈ x
        @test logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)

        d = Truncated(Normal(), 1, Inf)
        b = bijector(d)
        x = rand(d)
        y = b(x)
        @test y ≈ link(d, x)
        @test inv(b)(y) ≈ x
        @test logabsdetjac(b, x) ≈ logpdf_with_trans(d, x, false) - logpdf_with_trans(d, x, true)
    end

    @testset "Multivariate" begin
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
            @testset "$dist: dist" begin
                dist = Dirichlet([eps(Float64), 1000 * one(Float64)])
                td = transformed(dist)

                # single sample
                y = rand(td)
                x = inv(td.transform)(y)
                @test y == td.transform(x)
                @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

                # logpdf_with_jac
                lp, logjac = logpdf_with_jac(td, y)
                @test lp ≈ logpdf(td, y)
                @test logjac ≈ logabsdetjacinv(td.transform, y)

                # multi-sample
                y = rand(td, 10)
                x = inv(td.transform)(y)
                @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

                # forward
                f = forward(td)
                @test f.x ≈ inv(td.transform)(f.y)
                @test f.y ≈ td.transform(f.x)
                @test f.logabsdetjac ≈ logabsdetjac(td.transform, f.x)
                @test f.logpdf ≈ logpdf_with_trans(td.dist, f.x, true)

                # verify against AD
                # similar to what we do in test/transform.jl for Dirichlet
                if dist isa Dirichlet
                    b = Bijectors.SimplexBijector{Val{false}}()
                    x = rand(dist)
                    y = b(x)
                    @test log(abs(det(ForwardDiff.jacobian(b, x)))) ≈ logabsdetjac(b, x)
                    @test log(abs(det(ForwardDiff.jacobian(inv(b), y)))) ≈ logabsdetjac(inv(b), y)
                else
                    b = bijector(dist)
                    x = rand(dist)
                    y = b(x)
                    @test log(abs(det(ForwardDiff.jacobian(b, x)))) ≈ logabsdetjac(b, x)
                    @test log(abs(det(ForwardDiff.jacobian(inv(b), y)))) ≈ logabsdetjac(inv(b), y)
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
            InverseWishart(v,S)
        ]
        
        for dist in matrix_dists
            @testset "$dist: dist" begin
                td = transformed(dist)

                # single sample
                y = rand(td)
                x = inv(td.transform)(y)
                @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)

                # TODO: implement `logabsdetjac` for these
                # logpdf_with_jac
                # lp, logjac = logpdf_with_jac(td, y)
                # @test lp ≈ logpdf(td, y)
                # @test logjac ≈ logabsdetjacinv(td.transform, y)

                # multi-sample
                y = rand(td, 10)
                x = inv(td.transform)(y)
                @test logpdf(td, y) ≈ logpdf_with_trans(dist, x, true)
            end
        end
    end

    @testset "Composition <: Bijector" begin
        d = Beta()
        td = transformed(d)

        x = rand(d)
        y = td.transform(x)

        b = Bijectors.composel(td.transform, Bijectors.Identity{0}())
        ib = inv(b)

        @test forward(b, x) == forward(td.transform, x)
        @test forward(ib, y) == forward(inv(td.transform), y)

        @test forward(b, x) == forward(Bijectors.composer(b.ts...), x)

        # inverse works fine for composition
        cb = b ∘ ib
        @test cb(x) ≈ x

        cb2 = cb ∘ cb
        @test cb(x) ≈ x

        # ensures that the `logabsdetjac` is correct
        x = rand(d)
        b = inv(bijector(d))
        @test logabsdetjac(b ∘ b, x) ≈ logabsdetjac(b, b(x)) + logabsdetjac(b, x)

        # order of composed evaluation
        b1 = DistributionBijector(d)
        b2 = DistributionBijector(Gamma())

        cb = b1 ∘ b2
        @test cb(x) ≈ b1(b2(x))

        # contrived example
        b = bijector(d)
        cb = inv(b) ∘ b
        cb = cb ∘ cb
        @test (cb ∘ cb ∘ cb ∘ cb ∘ cb)(x) ≈ x

        # forward for tuple and array
        d = Beta()
        b = inv(bijector(d))
        b⁻¹ = inv(b)
        x = rand(d)

        cb_t = b⁻¹ ∘ b⁻¹
        f_t = forward(cb_t, x)

        cb_a = Composed([b⁻¹, b⁻¹])
        f_a = forward(cb_a, x)

        @test f_t == f_a

        # `composer` and `composel`
        cb_l = Bijectors.composel(b⁻¹, b⁻¹, b)
        cb_r = Bijectors.composer(reverse(cb_l.ts)...)
        y = cb_l(x)
        @test y == Bijectors.composel(cb_r.ts...)(x)

        k = length(cb_l.ts)
        @test all([cb_l.ts[i] == cb_r.ts[i] for i = 1:k])
    end

    @testset "Example: ADVI single" begin
        # Usage in ADVI
        d = Beta()
        b = DistributionBijector(d)    # [0, 1] → ℝ
        ib = inv(b)                    # ℝ → [0, 1]
        td = transformed(Normal(), ib) # x ∼ 𝓝(0, 1) then f(x) ∈ [0, 1]
        x = rand(td)                   # ∈ [0, 1]
        @test 0 ≤ x ≤ 1
    end
end
