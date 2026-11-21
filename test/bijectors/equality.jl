@testset "Equality" begin
    bs = [
        identity,
        elementwise(exp),
        elementwise(log),
        Scale(2.0),
        Scale(3.0),
        Scale(rand(2, 2)),
        Scale(rand(2, 2)),
        Shift(2.0),
        Shift(3.0),
        Shift(rand(2)),
        Shift(rand(2)),
        Logit(1.0, 2.0),
        Logit(1.0, 3.0),
        Logit(2.0, 3.0),
        Logit(0.0, 2.0),
        InvertibleBatchNorm(2),
        InvertibleBatchNorm(3),
        PDBijector(),
        Permute([1.0, 2.0, 3.0]),
        Permute([2.0, 3.0, 4.0]),
        PlanarLayer(2),
        PlanarLayer(3),
        RadialLayer(2),
        RadialLayer(3),
        SimplexBijector(),
        Stacked((elementwise(exp), elementwise(log))),
        Stacked((elementwise(log), elementwise(exp))),
        Stacked([elementwise(exp), elementwise(log)]),
        Stacked([elementwise(log), elementwise(exp)]),
        elementwise(exp) ∘ elementwise(log),
        elementwise(log) ∘ elementwise(exp),
        TruncatedBijector(1.0, 2.0),
        TruncatedBijector(1.0, 3.0),
        TruncatedBijector(0.0, 2.0),
    ]
    for i in eachindex(bs), j in eachindex(bs)
        if i == j
            @test bs[i] == deepcopy(bs[j])
            @test inverse(bs[i]) == inverse(deepcopy(bs[j]))
        else
            @test bs[i] != bs[j]
        end
    end
end

@testset "Hashing" begin
    VB = Bijectors.VectorBijectors
    h = hash(:bijector)
    for f in (
        Shift,
        Scale,
        x -> Bijectors.Inverse(Shift(x)),
        x -> VB.Elementwise(Shift(x), (2,)),
        x -> VB.ProductVecTransform([Shift(x)], [1:2], (2,)),
        x -> VB.ProductVecInvTransform([Shift(x)], [1:2], (2,)),
        x -> VB.ReshapeWrapper((1, 2), (2,), Shift(x)),
        x -> inverse(VB.ReshapeWrapper((1, 2), (2,), Shift(x))),
    )
        b1, b2 = f([1, 2]), f([1.0, 2.0])
        @test isequal(b1, b2)
        @test hash(b1, h) == hash(b2, h)
        @test get(Dict{Any,Int}(b1 => 1), b2, 0) == 1
    end
    for f in (Shift, Scale, x -> Bijectors.Inverse(Shift(x)))
        @test isequal(f(NaN), f(NaN))
        @test !isequal(f(0.0), f(-0.0))
    end
end
