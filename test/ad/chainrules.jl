@testset "chainrules" begin
    x, Δx, x̄ = randn(3)
    y, Δy, ȳ = randn(3)
    z, Δz, z̄ = randn(3)
    Δu = randn()

    ỹ = expm1(y)
    frule_test(Bijectors.find_alpha, (x, Δx), (ỹ, Δy), (z, Δz); rtol=1e-3, atol=1e-3)
    rrule_test(Bijectors.find_alpha, Δu, (x, x̄), (ỹ, ȳ), (z, z̄); rtol=1e-3, atol=1e-3)
end
