using Bijectors:
    Coupling,
    PartitionMask,
    coupling,
    couple,
    partition,
    combine,
    Shift,
    Scale

@testset "Coupling" begin
    @testset "PartitionMask" begin
        m1 = PartitionMask(3, [1], [2])
        m2 = PartitionMask(3, [1], [2], [3])

        @test (m1.A_1 == m2.A_1) & (m1.A_2 == m2.A_2) & (m1.A_3 == m2.A_3)

        x = [1., 2., 3.]
        x1, x2, x3 = partition(m1, x)
        @test (x1 == [1.]) & (x2 == [2.]) & (x3 == [3.])

        y = combine(m1, x1, x2, x3)
        @test y == x
    end

    @testset "Basics" begin
        m = PartitionMask(3, [1], [2])
        cl1 = Coupling(x -> Shift(x[1]), m)

        x = [1., 2., 3.]
        @test cl1(x) == [3., 2., 3.]

        cl2 = Coupling(θ -> Shift(θ[1]), m)
        @test cl2(x) == cl1(x)

        # inversion
        icl1 = inverse(cl1)
        @test icl1(cl1(x)) == x
        @test inverse(cl2)(cl2(x)) == x

        # This `cl2` should result in
        b = Shift(x[2:2])

        # logabsdetjac
        @test logabsdetjac(cl1, x) == logabsdetjac(b, x[1:1])

        # with_logabsdet_jacobian
        @test with_logabsdet_jacobian(cl1, x) == (cl1(x), logabsdetjac(cl1, x))
        @test with_logabsdet_jacobian(icl1, cl1(x)) == (x, - logabsdetjac(cl1, x))
    end

    @testset "Classic" begin
        m = PartitionMask(3, [1], [2])

        # With `Scale`
        cl = Coupling(x -> Scale(x[1]), m)
        x = [-1., -2., -3.]
        y = [2., -2., -3.]
        test_bijector(cl, x; y=y, logjac=log(2))

        x = [1., 2., 3.]
        y = [2., 2., 3.]
        test_bijector(cl, x; y=y, logjac=log(2))
    end
end
