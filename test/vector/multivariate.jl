const multivariates = [
    Multinomial(10, [0.2, 0.5, 0.3]),
    MvNormal([0.0, 0.0], I),
    MvNormalCanon([1.0, 2.0, 3.0], [4.0 -2.0 -1.0; -2.0 5.0 -1.0; -1.0 -1.0 6.0]),
    MvTDist(5.0, zeros(2), Matrix(1.0I, 2, 2)),
    MvTDist(1.0, [1.0, -1.0, 0.5], [2.0 0.5 0.0; 0.5 3.0 0.5; 0.0 0.5 1.5]),
    MvLogNormal([0.0, 0.0], I),
    MvLogitNormal([1.0, 2.0], Diagonal([4.0, 5.0])),
    Dirichlet([2.0, 3.0, 5.0]),
]

function _gen_testcases(::Val{:multivariates})
    cases = VectorTestCase[]
    for d in multivariates
        expected_zero_allocs = if d isa Union{Dirichlet,MvLogitNormal,MvLogNormal}
            (to_vec, from_vec)
        else
            (to_vec, from_vec, to_linked_vec, from_linked_vec)
        end
        push!(cases, VectorTestCase(d; expected_zero_allocs=expected_zero_allocs))
    end
    return cases
end
