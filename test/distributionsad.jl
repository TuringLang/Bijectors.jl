@testset "DistributionsAD" begin
    # https://github.com/TuringLang/Bijectors.jl/issues/298
    @testset "#298" begin
        dists = arraydist(fill(InverseGamma(), 2, 2))
        @test bijector(dists) isa Bijectors.TruncatedBijector
    end
end
