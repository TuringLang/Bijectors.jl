using Bijectors: ProductBijector
using FillArrays

has_square_jacobian(b, x) = Bijectors.output_size(b, x) == size(x)

@testset "ProductBijector" begin
    # Some distributions.
    ds = [
        # 1D.
        (Normal(), true),
        (InverseGamma(), false),
        (Beta(), false),
        # 2D.
        (MvNormal(Zeros(3), I), true),
        (Dirichlet(Ones(3)), false),
    ]

    # Stacking a single dimension.
    N = 4
    @testset "Single-dim stack: $(nameof(typeof(d)))" for (d, isidentity) in ds
        b = bijector(d)
        xs = [rand(d) for _ in 1:N]
        x = stack(xs)

        d_prod = product_distribution(Fill(d, N))
        b_prod = bijector(d_prod)

        sz_true = (Bijectors.output_size(b, size(xs[1]))..., N)
        @test Bijectors.output_size(b_prod, size(x)) == sz_true

        results = map(xs) do x
            with_logabsdet_jacobian(b, x)
        end
        y, logjac = stack(map(first, results)), sum(last, results)

        test_bijector(
            b_prod,
            x;
            y,
            logjac,
            changes_of_variables_test=has_square_jacobian(b, xs[1]),
            test_not_identity=(!isidentity),
        )
    end

    @testset "Two-dim stack: $(nameof(typeof(d)))" for (d, isidentity) in ds
        b = bijector(d)
        xs = [rand(d) for _ in 1:N, _ in 1:(N + 1)]
        x = stack(xs)

        d_prod = product_distribution(Fill(d, N, N + 1))
        b_prod = bijector(d_prod)

        sz_true = (Bijectors.output_size(b, size(xs[1]))..., N, N + 1)
        @test Bijectors.output_size(b_prod, size(x)) == sz_true

        results = map(Base.Fix1(with_logabsdet_jacobian, b), xs)
        y, logjac = stack(map(first, results)), sum(last, results)

        test_bijector(
            b_prod,
            x;
            y,
            logjac,
            changes_of_variables_test=has_square_jacobian(b, xs[1]),
            test_not_identity=(!isidentity),
        )
    end
end
