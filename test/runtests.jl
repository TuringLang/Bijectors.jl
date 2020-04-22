using ReverseDiff, Tracker, ForwardDiff, Zygote, DistributionsAD, Bijectors
using Random, LinearAlgebra, Combinatorics, Test
using DistributionsAD: TuringUniform, TuringMvNormal, TuringMvLogNormal, 
                        TuringPoissonBinomial

Random.seed!(123456)

function get_stage()
    if get(ENV, "TRAVIS", "") == "true" || get(ENV, "GITHUB_ACTIONS", "") == "true"
        if "STAGE" in keys(ENV)
            return ENV["STAGE"]
        else
            return "nonAD"
        end
    else
        if "STAGE" in keys(ENV)
            return ENV["STAGE"]
        else
            return "all"
        end
    end
end

stg = get_stage()
if stg in ("nonAD", "all")
    @testset "Interface" begin
        include("interface.jl")
    end
    include("transform.jl")
    include("norm_flows.jl")
    include("bijectors/permute.jl")
end
if stg != "nonAD"
    include("ad/ad_test_utils.jl")
    include("ad/distributions.jl")
end
