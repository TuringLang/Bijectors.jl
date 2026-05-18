@testset "test_inverse and test_with_logabsdet_jacobian" begin
    b = Bijectors.Scale{Float64}(4.2)
    x = 0.3

    InverseFunctions.test_inverse(b, x)
    ChangesOfVariables.test_with_logabsdet_jacobian(b, x, (f::Bijectors.Scale, x) -> f.a)
end
