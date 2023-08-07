@testset "AD for PD bijector" begin
    d = 4
    dist = Wishart(4, Matrix{Float64}(Distributions.I, d, d))
    x = rand(dist)
    b = bijector(dist)
    binv = inverse(b)
    y = b(x)

    test_ad(vec(x); use_forwarddiff_as_truth=true) do x
        sum(transform(b, reshape(x, d, d)))
    end

    test_ad(y; use_forwarddiff_as_truth=true) do y
        sum(transform(binv, y))
    end
end
