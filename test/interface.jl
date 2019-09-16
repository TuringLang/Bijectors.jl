using Test
using Bijectors
using Random
using LinearAlgebra
using ForwardDiff

Random.seed!(123)

struct NonInvertibleBijector{AD} <: ADBijector{AD} end

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
                b = DistributionBijector{Bijectors.ADBackend(:forward_diff), typeof(dist)}(dist)
                
                @test abs(det(Bijectors.jacobian(b, x))) > 0
                @test logabsdetjac(b, x) ≠ Inf

                y = b(x)
                b⁻¹ = inv(b)
                @test abs(det(Bijectors.jacobian(b⁻¹, y))) > 0
                @test logabsdetjac(b⁻¹, y) ≠ Inf
            end

            @testset "$dist: Tracker AD" begin
                x = rand(dist)
                b = DistributionBijector{Bijectors.ADBackend(:reverse_diff), typeof(dist)}(dist)
                
                @test abs(det(Bijectors.jacobian(b, x))) > 0
                @test logabsdetjac(b, x) ≠ Inf

                y = b(x)
                b⁻¹ = inv(b)
                @test abs(det(Bijectors.jacobian(b⁻¹, y))) > 0
                @test logabsdetjac(b⁻¹, y) ≠ Inf
            end
        end
    end

    @testset "Truncated" begin
        d = Truncated(Normal(), -1, 1)
        b = bijector(d)
        x = rand(d)
        @test b(x) == link(d, x)

        d = Truncated(Normal(), -Inf, 1)
        b = bijector(d)
        x = rand(d)
        @test b(x) == link(d, x)

        d = Truncated(Normal(), 1, Inf)
        b = bijector(d)
        x = rand(d)
        @test b(x) == link(d, x)
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

        b = Bijectors.composel(td.transform, Bijectors.Identity())
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

    @testset "Stacked <: Bijector" begin
        # `logabsdetjac` without AD
        d = Beta()
        b = bijector(d)
        x = rand(d)
        y = b(x)
        sb = vcat(b, b, inv(b), inv(b))
        @test logabsdetjac(sb, [x, x, y, y]) ≈ 0.0

        # `logabsdetjac` with AD
        b = DistributionBijector(d)
        y = b(x)
        sb1 = vcat(b, b, inv(b), inv(b))             # <= tuple
        sb2 = Stacked([b, b, inv(b), inv(b)])        # <= Array
        @test logabsdetjac(sb1, [x, x, y, y]) ≈ 0.0
        @test logabsdetjac(sb2, [x, x, y, y]) ≈ 0.0

        # value-test
        x = ones(3)
        sb = vcat(Bijectors.Exp(), Bijectors.Log(), Bijectors.Shift(5.0))
        @test sb(x) == [exp(x[1]), log(x[2]), x[3] + 5.0]
        @test logabsdetjac(sb, x) == sum([logabsdetjac(sb.bs[i], x[i]) for i = 1:3])

        # TODO: change when we have dimensionality in the type
        sb = vcat([Bijectors.Exp(), Bijectors.SimplexBijector()]...)
        @test_throws AssertionError sb(x ./ sum(x))

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
                MvNormal(zeros(2), ones(2))
            ]

            ranges = []
            idx = 1
            for i = 1:length(dists)
                d = dists[i]
                push!(ranges, idx:idx + length(d) - 1)
                idx += length(d)
            end

            num_params = ranges[end][end]
            d = MvNormal(zeros(num_params), ones(num_params))

            # Stacked{<:Array}
            bs = bijector.(dists)     # constrained-to-unconstrained bijectors for dists
            ibs = inv.(bs)            # invert, so we get unconstrained-to-constrained
            sb = Stacked(ibs, ranges) # => Stacked <: Bijector
            x = rand(d)
            sb(x)
            @test sb isa Stacked

            td = transformed(d, sb)  # => MultivariateTransformed <: Distribution{Multivariate, Continuous}
            @test td isa Distribution{Multivariate, Continuous}

            # check that wrong ranges fails
            sb = vcat(ibs...)
            td = transformed(d, sb)
            x = rand(d)
            @test_throws AssertionError sb(x)

            # Stacked{<:Tuple}
            bs = bijector.(tuple(dists...))
            ibs = inv.(bs)
            sb = Stacked(ibs, ranges)
            isb = inv(sb)
            @test sb isa Stacked{<: Tuple}

            # inverse
            td = transformed(d, sb)
            y = rand(td)
            x = isb(y)
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
        end
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
