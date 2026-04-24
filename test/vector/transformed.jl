module VBTransformedTests

using Bijectors
using Distributions
using LinearAlgebra
using Test
using Bijectors.VectorBijectors
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

transformed_dists = [
    # Univariate
    transformed(Normal(), exp),
    transformed(Beta(2, 3), Bijectors.Logit(0.0, 1.0)),
    transformed(Gamma(2, 1), Bijectors.Log{0}()),
    # Multivariate
    transformed(product_distribution(fill(Normal(), 4)), elementwise(exp)),
    transformed(MvNormal(zeros(3), I), Bijectors.Scale(2.0)),
    transformed(Dirichlet([1.0, 2.0, 3.0])),
    transformed(MvLogNormal(zeros(2), I), elementwise(log)),
]

@testset "TransformedDistributions" begin
    for d in transformed_dists
        VectorBijectors.test_all(d; test_in_support=false)
    end
end

end # module VBTransformedTests
