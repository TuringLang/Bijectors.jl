using Bijectors: Reshape

@testset "Reshape" begin
    dist = reshape(product_distribution(fill(InverseGamma(2, 3), 10)), 2, 5)
    b = bijector(dist)

    x = rand(dist)
    test_bijector(b, x)
end
