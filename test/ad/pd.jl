_topd(x) = x * x' + I

@testset "AD for PDVecBijector" begin
    d = 4
    b = Bijectors.PDVecBijector()
    binv = inverse(b)

    z = randn(d, d)
    x = _topd(z)
    y = b(x)

    test_ad(vec(z)) do x
        sum(transform(b, _topd(reshape(x, d, d))))
    end

    test_ad(y) do y
        sum(transform(binv, y))
    end
end
