# `VectorBijectors.CDF` / `Quantile` are not registered as `scalar_to_scalar_bijector`
# for any distribution, so the framework-level `VectorBijectors.test_all` sweep does not
# cover them. Mirroring the `test/vector/` convention, we separate cases (data) from the
# runner here.

const _CDF_QUANTILE_VEC_CASES = [
    (Bijectors.VectorBijectors.CDF(Normal()), 0.3),
    (Bijectors.VectorBijectors.CDF(Gamma(2, 3)), 1.7),
    (Bijectors.VectorBijectors.Quantile(Normal()), 0.3),
    (Bijectors.VectorBijectors.Quantile(Gamma(2, 3)), 0.3),
]

function _expected_wladj(b::Bijectors.VectorBijectors.CDF, x)
    return cdf(b.dist, x), logpdf(b.dist, x)
end
function _expected_wladj(q::Bijectors.VectorBijectors.Quantile, x)
    y = quantile(q.dist, x)
    return y, -logpdf(q.dist, y)
end

function run_cdf_quantile_vec_case(b, x; tol=1.0e-9)
    expected_y, expected_ladj = _expected_wladj(b, x)
    y, ladj = with_logabsdet_jacobian(b, x)
    @test y ≈ expected_y
    @test ladj ≈ expected_ladj
    ib = inverse(b)
    @test inverse(ib) isa typeof(b)
    x_back, ladj_inv = with_logabsdet_jacobian(ib, y)
    @test x_back ≈ x
    @test ladj + ladj_inv ≈ 0 atol = tol
    @test Bijectors.is_monotonically_increasing(b)
    @test Bijectors.is_monotonically_increasing(ib)
    return nothing
end

@testset "VectorBijectors.CDF / VectorBijectors.Quantile" begin
    @testset "$(typeof(b)) at x=$x" for (b, x) in _CDF_QUANTILE_VEC_CASES
        run_cdf_quantile_vec_case(b, x)
    end
end
