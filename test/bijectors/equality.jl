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
    product = product_distribution([Normal(), Beta(2, 2)])
    reshaped = VB.ReshapeWrapper((1, 1), (1,), Shift([1.0]))
    for b in (
        Shift([1.0, 2.0]),
        Scale([1.0, 2.0]),
        Bijectors.Inverse(Shift([1.0])),
        VB.to_linked_vec(product),
        VB.from_linked_vec(product),
        reshaped,
        inverse(reshaped),
    )
        copy = deepcopy(b)
        @test isequal(b, copy)
        @test hash(b) == hash(copy)
        @test get(Dict{Any,Int}(b => 1), copy, 0) == 1
    end
    @test hash(VB.Elementwise(Shift(1), (2,))) == hash(VB.Elementwise(Shift(1.0), (2,)))
    for f in (Shift, Scale, x -> Bijectors.Inverse(Shift(x)))
        @test isequal(f(NaN), f(NaN))
        @test !isequal(f(0.0), f(-0.0))
    end
end
