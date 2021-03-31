@testset "chainrules" begin
    x = randn()
    y = expm1(randn())
    z = randn()
    test_frule(Bijectors.find_alpha, x, y, z)
    test_rrule(Bijectors.find_alpha, x, y, z)
end
