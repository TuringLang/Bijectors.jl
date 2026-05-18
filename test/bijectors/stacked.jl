struct ProjectionBijector <: Bijectors.Bijector end

Bijectors.output_size(::ProjectionBijector, sz::Tuple{Int}) = (sz[1] - 1,)
Bijectors.output_size(::Inverse{ProjectionBijector}, sz::Int) = (sz[1] + 1,)

function Bijectors.with_logabsdet_jacobian(::ProjectionBijector, x::AbstractVector)
    return x[1:(end - 1)], 0
end
function Bijectors.with_logabsdet_jacobian(::Inverse{ProjectionBijector}, x::AbstractVector)
    return vcat(x, 0), 0
end

@testset "Stacked with differing input and output size" begin
    bs = [
        Stacked((elementwise(exp), ProjectionBijector()), (1:1, 2:3)),
        Stacked([elementwise(exp), ProjectionBijector()], [1:1, 2:3]),
        Stacked([elementwise(exp), ProjectionBijector()], (1:1, 2:3)),
        Stacked((elementwise(exp), ProjectionBijector()), [1:1, 2:3]),
    ]
    @testset "$b" for b in bs
        binv = inverse(b)
        x = [1.0, 2.0, 3.0]
        y = b(x)
        x_ = binv(y)

        # Are the values of correct size?
        @test size(y) == (2,)
        @test size(x_) == (3,)
        # Can we determine the sizes correctly?
        @test Bijectors.output_size(b, size(x)) == (2,)
        @test Bijectors.output_size(binv, size(y)) == (3,)

        # Are values correct?
        @test y == [exp(1.0), 2.0]
        @test binv(y) == [1.0, 2.0, 0.0]
    end

    @testset "composition" begin
        # Composition with one dimension reduction.
        b = Stacked((elementwise(exp), ProjectionBijector() ∘ identity), [1:1, 2:3])
        binv = inverse(b)
        x = [1.0, 2.0, 3.0]
        y = b(x)
        x_ = binv(y)

        # Are the values of correct size?
        @test size(y) == (2,)
        @test size(x_) == (3,)
        # Can we determine the sizes correctly?
        @test Bijectors.output_size(b, size(x)) == (2,)
        @test Bijectors.output_size(binv, size(y)) == (3,)

        # Are values correct?
        @test y == [exp(1.0), 2.0]
        @test binv(y) == [1.0, 2.0, 0.0]

        # Composition with two dimension reductions.
        b = Stacked(
            (elementwise(exp), ProjectionBijector() ∘ ProjectionBijector()), [1:1, 2:4]
        )
        binv = inverse(b)
        x = [1.0, 2.0, 3.0, 4.0]
        y = b(x)
        x_ = binv(y)

        # Are the values of correct size?
        @test size(y) == (2,)
        @test size(x_) == (4,)
        # Can we determine the sizes correctly?
        @test Bijectors.output_size(b, size(x)) == (2,)
        @test Bijectors.output_size(binv, size(y)) == (4,)

        # Are values correct?
        @test y == [exp(1.0), 2.0]
        @test binv(y) == [1.0, 2.0, 0.0, 0.0]
    end
end

@testset "Stacked <: Bijector" begin
    # `logabsdetjac` withOUT AD
    d = Beta()
    b = bijector(d)
    x = rand(d)
    y = b(x)

    sb1 = @inferred Stacked(b, b, inverse(b), inverse(b))             # <= Tuple
    res1 = with_logabsdet_jacobian(sb1, [x, x, y, y])

    @test sb1([x, x, y, y]) ≈ res1[1]
    @test logabsdetjac(sb1, [x, x, y, y]) ≈ 0 atol = 1e-6
    @test res1[2] ≈ 0 atol = 1e-6

    sb2 = Stacked([b, b, inverse(b), inverse(b)])        # <= Array
    res2 = with_logabsdet_jacobian(sb2, [x, x, y, y])

    @test sb2([x, x, y, y]) ≈ res2[1]
    @test logabsdetjac(sb2, [x, x, y, y]) ≈ 0.0 atol = 1e-12
    @test res2[2] ≈ 0.0 atol = 1e-12

    # value-test
    x = ones(3)
    sb = @inferred Stacked(elementwise(exp), elementwise(log), Shift(5.0))
    res = with_logabsdet_jacobian(sb, x)
    @test sb(x) == [exp(x[1]), log(x[2]), x[3] + 5.0]
    @test res[1] == [exp(x[1]), log(x[2]), x[3] + 5.0]
    @test logabsdetjac(sb, x) ==
        sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges_in[i]])) for i in 1:3])
    @test res[2] == logabsdetjac(sb, x)

    # TODO: change when we have dimensionality in the type
    sb = @inferred Stacked((elementwise(exp), SimplexBijector()), (1:1, 2:3))
    x = ones(3) ./ 3.0
    res = @inferred with_logabsdet_jacobian(sb, x)
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) ==
        sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges_in[i]])) for i in 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws ErrorException sb(x)

    # Array-version
    sb = Stacked([elementwise(exp), SimplexBijector()], [1:1, 2:3])
    x = ones(3) ./ 3.0
    res = with_logabsdet_jacobian(sb, x)
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) ==
        sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges_in[i]])) for i in 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws ErrorException sb(x)

    # Mixed versions
    # Tuple, Array
    sb = Stacked([elementwise(exp), SimplexBijector()], (1:1, 2:3))
    x = ones(3) ./ 3.0
    res = with_logabsdet_jacobian(sb, x)
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) ==
        sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges_in[i]])) for i in 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws ErrorException sb(x)

    # Array, Tuple
    sb = Stacked((elementwise(exp), SimplexBijector()), [1:1, 2:3])
    x = ones(3) ./ 3.0
    res = with_logabsdet_jacobian(sb, x)
    @test sb(x) == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test res[1] == [exp(x[1]), sb.bs[2](x[2:3])...]
    @test logabsdetjac(sb, x) ==
        sum([sum(logabsdetjac(sb.bs[i], x[sb.ranges_in[i]])) for i in 1:2])
    @test res[2] == logabsdetjac(sb, x)

    x = ones(4) ./ 4.0
    @test_throws ErrorException sb(x)

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
            MvNormal(zeros(2), I),
        ]

        ranges = []
        idx = 1
        for i in 1:length(dists)
            d = dists[i]
            push!(ranges, idx:(idx + length(d) - 1))
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
        @test td isa Distribution{Multivariate,Continuous}

        # check that wrong ranges fails
        sb = Stacked(ibs)
        x = rand(d)
        @test_throws ErrorException sb(x)

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
        y_ = vcat([ibs[i](x[ranges[i]]) for i in 1:length(dists)]...)
        x_ = vcat([bs[i](y[ranges[i]]) for i in 1:length(dists)]...)
        @test x ≈ x_
        @test y ≈ y_

        # AD verification
        @test log(abs(det(ForwardDiff.jacobian(sb, x)))) ≈ logabsdetjac(sb, x)
        @test log(abs(det(ForwardDiff.jacobian(isb, y)))) ≈ logabsdetjac(isb, y)

        # Ensure `Stacked` works for a single bijector
        d = (MvNormal(zeros(2), I),)
        sb = Stacked(bijector.(d), (1:2,))
        x = [0.5, 1.0]
        @test sb(x) == x
        @test logabsdetjac(sb, x) == 0
        @test with_logabsdet_jacobian(sb, x) == (x, zero(eltype(x)))
    end
end
