using Bijectors: CDFBijector, QuantileBijector

@testset "CDFBijector / QuantileBijector" begin
    # `test_bijector` calls `getjacobian`, which routes through ForwardDiff.
    # `Distributions.cdf(::Gamma, ::Dual)` is not differentiable, so AD-dependent
    # checks are skipped for Gamma.
    @testset "test_bijector: $(B)($(nameof(typeof(dist))))" for dist in
                                                                (Normal(), Gamma(2, 3)),
        B in (CDFBijector, QuantileBijector)

        b = B(dist)
        x = 0.3
        if dist isa Normal
            test_bijector(b, x; test_not_identity=false)
            B === CDFBijector && test_bijector(b, [0.1, 0.4, 0.7]; test_not_identity=false)
        else
            test_bijector(
                b,
                x;
                test_not_identity=false,
                changes_of_variables_test=false,
                inverse_functions_test=false,
            )
        end
    end

    @testset "semantics" begin
        d = Normal()
        @test transform(CDFBijector(d), 0.3) ≈ cdf(d, 0.3)
        @test transform(QuantileBijector(d), 0.3) ≈ quantile(d, 0.3)
        @test inverse(CDFBijector(d)) == QuantileBijector(d)
        @test inverse(QuantileBijector(d)) == CDFBijector(d)
        # broadcast convention: logabsdetjac sums to a scalar (cf. Logit/Scale)
        xs = [-1.5, -0.4, 0.0, 0.7, 1.8]
        @test logabsdetjac(CDFBijector(d), xs) isa Real
        @test logabsdetjac(CDFBijector(d), xs) ≈
            sum(logabsdetjac(CDFBijector(d), x) for x in xs)
    end

    @testset "monotonicity" begin
        @test Bijectors.is_monotonically_increasing(CDFBijector(Normal()))
        @test Bijectors.is_monotonically_increasing(QuantileBijector(Gamma()))
    end

    @testset "equality and show" begin
        @test CDFBijector(Normal()) == CDFBijector(Normal())
        @test CDFBijector(Normal()) != CDFBijector(Normal(0, 2))
        @test CDFBijector(Normal()) != QuantileBijector(Normal())
        @test occursin(r"^CDFBijector\(.*\)$", sprint(show, CDFBijector(Normal())))
        @test occursin(r"^QuantileBijector\(.*\)$", sprint(show, QuantileBijector(Gamma())))
    end
end
