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
    for i in 1:length(bs), j in 1:length(bs)
        if i == j
            @test bs[i] == deepcopy(bs[j])
            @test inverse(bs[i]) == inverse(deepcopy(bs[j]))
        else
            @test bs[i] != bs[j]
        end
    end
end
