@testset "chainrules" begin
    x = randn()
    y = expm1(randn())
    z = randn()
    test_frule(Bijectors.find_alpha, x, y, z)
    test_rrule(Bijectors.find_alpha, x, y, z)

    test_rrule(Bijectors.combine, Bijectors.PartitionMask(3, [1], [2]) ⊢ ChainRulesTestUtils.NoTangent(), [1.0], [2.0], [3.0])

    # ordered bijector
    b = Bijectors.OrderedBijector()
    test_rrule(Bijectors._transform_ordered, randn(5))
    test_rrule(Bijectors._transform_ordered, randn(5, 2))
    test_rrule(Bijectors._transform_inverse_ordered, b(rand(5)))
    test_rrule(Bijectors._transform_inverse_ordered, b(rand(5, 2)))
end
