@testset "chainrules" begin
    x = randn()
    y = expm1(randn())
    z = randn()
    test_frule(Bijectors.find_alpha, x, y, z)
    test_rrule(Bijectors.find_alpha, x, y, z)

    if @isdefined Tapir
        Tapir.TestUtils.test_rrule!!(
            Xoshiro(123), Bijectors.find_alpha, x, y, z;
            is_primitive=true, perf_flag=:none,
        )
        Tapir.TestUtils.test_rrule!!(
            Xoshiro(123), Bijectors.find_alpha, x, y, 3;
            is_primitive=true, perf_flag=:none,
        )
        Tapir.TestUtils.test_rrule!!(
            Xoshiro(123), Bijectors.find_alpha, x, y, UInt32(3);
            is_primitive=true, perf_flag=:none,
        )
    end

    test_rrule(
        Bijectors.combine,
        Bijectors.PartitionMask(3, [1], [2]) ⊢ ChainRulesTestUtils.NoTangent(),
        [1.0],
        [2.0],
        [3.0],
    )

    # ordered bijector
    b = Bijectors.OrderedBijector()
    test_rrule(Bijectors._transform_ordered, randn(5))
    test_rrule(Bijectors._transform_ordered, randn(5, 2))
    test_rrule(Bijectors._transform_inverse_ordered, b(rand(5)))
    test_rrule(Bijectors._transform_inverse_ordered, b(rand(5, 2)))

    # LKJ and LKJCholesky bijector
    dist = LKJCholesky(3, 4)
    x = rand(dist)
    test_rrule(Bijectors._link_chol_lkj_from_upper, x.U)
    test_rrule(Bijectors._link_chol_lkj_from_lower, x.L)

    b = bijector(dist)
    y = b(x)
    test_rrule(Bijectors._inv_link_chol_lkj, y)
end
