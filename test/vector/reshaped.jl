const reshaped_default_dists = [
    # 0-dim array output is blocked by
    # https://github.com/JuliaStats/Distributions.jl/issues/2025
    # reshape(Normal(), ()),
    vec(Normal()),
    reshape(Normal(), (1, 1, 1, 1, 1)),
    vec(Beta(2, 2)),
    vec(Poisson(3)),
    reshape(Poisson(3), (1, 1, 1, 1, 1)),
    reshape(MvNormal(zeros(2), I), (2, 1, 1)),
    reshape(MvNormal(zeros(4), I), (2, 2)),
    reshape(Dirichlet(ones(6)), (2, 3)),
    reshape(MatrixNormal(2, 4), 8),
    reshape(MatrixNormal(2, 5), 5, 2),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 16),
    reshape(Wishart(7, Matrix{Float64}(I, 4, 4)), 1, 1, 4, 1, 4),
]

function _gen_testcases(::Val{:reshaped_dists})
    return [VectorTestCase(d; expected_zero_allocs=()) for d in reshaped_default_dists]
end

# `reshape(Beta(2, 2), (1, 1, 1, 1, 1))` hit
# https://github.com/EnzymeAD/Enzyme.jl/issues/2987 on Julia 1.10 — Enzyme Reverse fails
# there, so callers may need a smaller adtype list for this one case.
const reshaped_beta_dist = reshape(Beta(2, 2), (1, 1, 1, 1, 1))

function _gen_testcases(::Val{:reshaped_beta_special})
    return [VectorTestCase(reshaped_beta_dist; expected_zero_allocs=())]
end
