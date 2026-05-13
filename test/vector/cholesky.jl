@testset "Cholesky" begin
    # Enzyme is tested separately in test/integration/enzyme.
    adtypes = [
        AutoReverseDiff(),
        AutoReverseDiff(; compile=true),
        AutoMooncake(),
        AutoMooncakeForward(),
    ]
    for c in generate_testcases(Val(:cholesky_dists))
        run_vector_case(c, adtypes)
    end
end
