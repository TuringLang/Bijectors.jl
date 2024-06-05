@testset "AD for StackedBijector" begin
    dist1 = Dirichlet(4, 1.0)
    b1 = bijector(dist1)

    dist2 = LogNormal(0.0, 1.0)
    b2 = bijector(dist2)

    x1 = rand(dist1)
    x2 = rand(dist2)
    
    y1 = b1(x1)
    y2 = b2(x2)

    b = Stacked((b1, b2), (1:4, 5:5))
    binv = inverse(b)

    y = vcat(y1, [y2])
    x = binv(y)

    test_ad(y) do x
        sum(transform(b, binv(x)))
    end

    test_ad(y) do y
        sum(transform(binv, y))
    end
end
