@testset "distributions" begin
    Random.seed!(1234)

    # Create random vectors and matrices
    dim = 3
    a = rand(dim)
    b = rand(dim)
    c = rand(dim)
    A = rand(dim, dim)
    B = rand(dim, dim)
    C = rand(dim, dim)

    dim_big = 10

    # Some LKJ problems may be hidden when test matrix is too small
    A_big = rand(dim_big, dim_big) 
    B_big = rand(dim_big, dim_big)

    # Create a random number
    alpha = rand()

    # Create positive definite matrix
    to_posdef(A::AbstractMatrix) = A * A' + I
    to_posdef_diagonal(a::AbstractVector) = Diagonal(a.^2 .+ 1)

    # Define adjoints for Tracker
    to_posdef(A::TrackedMatrix) = Tracker.track(to_posdef, A)
    Tracker.@grad function to_posdef(A::TrackedMatrix)
        data_A = Tracker.data(A)
        S = data_A * data_A' + I
        function pullback(∇)
            return ((∇ + ∇') * data_A,)
        end
        return S, pullback
    end

    # Create matrix `X` such that `X` and `I - X` are positive definite
    function to_beta_mat(A)
        S = A * A' + I
        invL = inv(cholesky(S).L)
        return invL * invL'
    end

    function to_corr(x)
        y = to_posdef(x)
        d = 1 ./ sqrt.(diag(y))
        y2 = d .* y .* d'
        return (y2 + y2') / 2
    end

    univariate_distributions = DistSpec[
        ## Univariate discrete distributions

        DistSpec(Bernoulli, (0.45,), 1),
        DistSpec(Bernoulli, (0.45,), [1, 1]),
        DistSpec(Bernoulli, (0.45,), 0),
        DistSpec(Bernoulli, (0.45,), [0, 0]),

        DistSpec((a, b) -> BetaBinomial(10, a, b), (2.0, 1.0), 5),
        DistSpec((a, b) -> BetaBinomial(10, a, b), (2.0, 1.0), [5, 5]),

        DistSpec(p -> Binomial(10, p), (0.5,), 5),
        DistSpec(p -> Binomial(10, p), (0.5,), [5, 5]),

        DistSpec(p -> Categorical(p / sum(p)), ([0.45, 0.55],), 1),
        DistSpec(p -> Categorical(p / sum(p)), ([0.45, 0.55],), [1, 1]),

        DistSpec(Geometric, (0.45,), 3),
        DistSpec(Geometric, (0.45,), [3, 3]),

        DistSpec(NegativeBinomial, (3.5, 0.5), 1),
        DistSpec(NegativeBinomial, (3.5, 0.5), [1, 1]),

        DistSpec(Poisson, (0.5,), 1),
        DistSpec(Poisson, (0.5,), [1, 1]),

        DistSpec(Skellam, (1.0, 2.0), -2),
        DistSpec(Skellam, (1.0, 2.0), [-2, -2]),

        DistSpec(PoissonBinomial, ([0.5, 0.5],), 0),
        DistSpec(PoissonBinomial, ([0.5, 0.5],), [0, 0]),

        DistSpec(TuringPoissonBinomial, ([0.5, 0.5],), 0),
        DistSpec(TuringPoissonBinomial, ([0.5, 0.5],), [0, 0]),

        ## Univariate continuous distributions

        DistSpec(Arcsine, (), 0.5),
        DistSpec(Arcsine, (1.0,), 0.5),
        DistSpec(Arcsine, (0.0, 2.0), 0.5),

        DistSpec(Beta, (), 0.5),
        DistSpec(Beta, (1.0,), 0.5),
        DistSpec(Beta, (1.0, 2.0), 0.5),

        DistSpec(BetaPrime, (), 0.5),
        DistSpec(BetaPrime, (1.0,), 0.5),
        DistSpec(BetaPrime, (1.0, 2.0), 0.5),

        DistSpec(Biweight, (), 0.5),
        DistSpec(Biweight, (1.0,), 0.5),
        DistSpec(Biweight, (1.0, 2.0), 0.5),

        DistSpec(Cauchy, (), 0.5),
        DistSpec(Cauchy, (1.0,), 0.5),
        DistSpec(Cauchy, (1.0, 2.0), 0.5),

        DistSpec(Chi, (1.0,), 0.5),

        DistSpec(Chisq, (1.0,), 0.5),

        DistSpec(Cosine, (1.0, 1.0), 0.5),

        DistSpec(Epanechnikov, (1.0, 1.0), 0.5),

        DistSpec(s -> Erlang(1, s), (1.0,), 0.5), # First arg is integer

        DistSpec(Exponential, (1.0,), 0.5),

        DistSpec(FDist, (1.0, 1.0), 0.5),

        DistSpec(Frechet, (), 0.5),
        DistSpec(Frechet, (1.0,), 0.5),
        DistSpec(Frechet, (1.0, 2.0), 0.5),

        DistSpec(Gamma, (), 0.5),
        DistSpec(Gamma, (1.0,), 0.5),
        DistSpec(Gamma, (1.0, 2.0), 0.5),

        DistSpec(GeneralizedExtremeValue, (1.0, 1.0, 1.0), 0.5),

        DistSpec(GeneralizedPareto, (), 0.5),
        DistSpec(GeneralizedPareto, (1.0, 2.0), 0.5),
        DistSpec(GeneralizedPareto, (0.0, 2.0, 3.0), 0.5),

        DistSpec(Gumbel, (), 0.5),
        DistSpec(Gumbel, (1.0,), 0.5),
        DistSpec(Gumbel, (1.0, 2.0), 0.5),

        DistSpec(InverseGamma, (), 0.5),
        DistSpec(InverseGamma, (1.0,), 0.5),
        DistSpec(InverseGamma, (1.0, 2.0), 0.5),

        DistSpec(InverseGaussian, (), 0.5),
        DistSpec(InverseGaussian, (1.0,), 0.5),
        DistSpec(InverseGaussian, (1.0, 2.0), 0.5),

        DistSpec(Kolmogorov, (), 0.5),

        DistSpec(Laplace, (), 0.5),
        DistSpec(Laplace, (1.0,), 0.5),
        DistSpec(Laplace, (1.0, 2.0), 0.5),

        DistSpec(Levy, (), 0.5),
        DistSpec(Levy, (0.0,), 0.5),
        DistSpec(Levy, (0.0, 2.0), 0.5),

        DistSpec((a, b) -> LocationScale(a, b, Normal()), (1.0, 2.0), 0.5),

        DistSpec(Logistic, (), 0.5),
        DistSpec(Logistic, (1.0,), 0.5),
        DistSpec(Logistic, (1.0, 2.0), 0.5),

        DistSpec(LogitNormal, (), 0.5),
        DistSpec(LogitNormal, (1.0,), 0.5),
        DistSpec(LogitNormal, (1.0, 2.0), 0.5),

        DistSpec(LogNormal, (), 0.5),
        DistSpec(LogNormal, (1.0,), 0.5),
        DistSpec(LogNormal, (1.0, 2.0), 0.5),

        DistSpec(Normal, (), 0.5),
        DistSpec(Normal, (1.0,), 0.5),
        DistSpec(Normal, (1.0, 2.0), 0.5),

        DistSpec(NormalCanon, (1.0, 2.0), 0.5),

        DistSpec(NormalInverseGaussian, (1.0, 2.0, 1.0, 1.0), 0.5),

        DistSpec(Pareto, (), 1.5),
        DistSpec(Pareto, (1.0,), 1.5),
        DistSpec(Pareto, (1.0, 1.0), 1.5),

        DistSpec(PGeneralizedGaussian, (), 0.5),
        DistSpec(PGeneralizedGaussian, (1.0, 1.0, 1.0), 0.5),

        DistSpec(Rayleigh, (), 0.5),
        DistSpec(Rayleigh, (1.0,), 0.5),

        DistSpec(Semicircle, (1.0,), 0.5),

        DistSpec(SymTriangularDist, (), 0.5),
        DistSpec(SymTriangularDist, (1.0,), 0.5),
        DistSpec(SymTriangularDist, (1.0, 2.0), 0.5),

        DistSpec(TDist, (1.0,), 0.5),

        DistSpec(TriangularDist, (1.0, 3.0), 1.5),
        DistSpec(TriangularDist, (1.0, 3.0, 2.0), 1.5),

        DistSpec(Triweight, (1.0, 1.0), 1.0),

        DistSpec(
            (mu, sigma, l, u) -> truncated(Normal(mu, sigma), l, u), (0.0, 1.0, 1.0, 2.0), 1.5
        ),

        DistSpec(Uniform, (), 0.5),
        DistSpec(Uniform, (0.0, 1.0), 0.5),

        DistSpec(TuringUniform, (), 0.5),
        DistSpec(TuringUniform, (0.0, 1.0), 0.5),

        DistSpec(VonMises, (), 1.0),

        DistSpec(Weibull, (), 1.0),
        DistSpec(Weibull, (1.0,), 1.0),
        DistSpec(Weibull, (1.0, 1.0), 1.0),
    ]

    broken_univariate_distributions = DistSpec[
        # Zygote
        DistSpec(Chernoff, (), 0.5),

        # Broken in Distributions even without autodiff
        DistSpec(() -> KSDist(1), (), 0.5),
        DistSpec(() -> KSOneSided(1), (), 0.5),
        DistSpec(StudentizedRange, (1.0, 2.0), 0.5),

        # Dispatch error caused by ccall
        DistSpec(NoncentralBeta, (1.0, 2.0, 1.0), 0.5),
        DistSpec(NoncentralChisq, (1.0, 2.0), 0.5),
        DistSpec(NoncentralF, (1, 2, 1), 0.5),
        DistSpec(NoncentralT, (1, 2), 0.5),

        # Stackoverflow caused by SpecialFunctions.besselix
        DistSpec(VonMises, (1.0,), 1.0),
        DistSpec(VonMises, (1, 1), 1),
    ]

    multivariate_distributions = DistSpec[
        ## Multivariate discrete distributions

        # Vector x
        DistSpec(p -> Multinomial(2, p ./ sum(p)), (fill(0.5, 2),), [2, 0]),

        # Vector x
        DistSpec((m, A) -> MvNormal(m, to_posdef(A)), (a, A), b),
        DistSpec(MvNormal, (a, b), c),
        DistSpec((m, s) -> MvNormal(m, to_posdef_diagonal(s)), (a, b), c),
        DistSpec(MvNormal, (a, alpha), b),
        DistSpec((m, s) -> MvNormal(m, s^2 * I), (a, alpha), b),
        DistSpec(A -> MvNormal(to_posdef(A)), (A,), a),
        DistSpec(MvNormal, (a,), b),
        DistSpec(s -> MvNormal(to_posdef_diagonal(s)), (a,), b),
        DistSpec(s -> MvNormal(dim, s), (alpha,), a),
        DistSpec((m, A) -> TuringMvNormal(m, to_posdef(A)), (a, A), b),
        DistSpec(TuringMvNormal, (a, b), c),
        DistSpec((m, s) -> TuringMvNormal(m, to_posdef_diagonal(s)), (a, b), c),
        DistSpec(TuringMvNormal, (a, alpha), b),
        DistSpec((m, s) -> TuringMvNormal(m, s^2 * I), (a, alpha), b),
        DistSpec(A -> TuringMvNormal(to_posdef(A)), (A,), a),
        DistSpec(TuringMvNormal, (a,), b),
        DistSpec(s -> TuringMvNormal(to_posdef_diagonal(s)), (a,), b),
        DistSpec(s -> TuringMvNormal(dim, s), (alpha,), a),
        DistSpec((m, A) -> MvLogNormal(m, to_posdef(A)), (a, A), b),
        DistSpec(MvLogNormal, (a, b), c),
        DistSpec((m, s) -> MvLogNormal(m, to_posdef_diagonal(s)), (a, b), c),
        DistSpec(MvLogNormal, (a, alpha), b),
        DistSpec(A -> MvLogNormal(to_posdef(A)), (A,), a),
        DistSpec(MvLogNormal, (a,), b),
        DistSpec(s -> MvLogNormal(to_posdef_diagonal(s)), (a,), b),
        DistSpec(s -> MvLogNormal(dim, s), (alpha,), a),

        DistSpec(Dirichlet, (ones(dim),), b ./ sum(b)),

        # Matrix case
        DistSpec(MvNormal, (a, b), A),
        DistSpec((m, s) -> MvNormal(m, to_posdef_diagonal(s)), (a, b), A),
        DistSpec(MvNormal, (a, alpha), A),
        DistSpec((m, s) -> MvNormal(m, s^2 * I), (a, alpha), A),
        DistSpec(MvNormal, (a,), A),
        DistSpec(s -> MvNormal(to_posdef_diagonal(s)), (a,), A),
        DistSpec(s -> MvNormal(dim, s), (alpha,), A),
        DistSpec(MvLogNormal, (a, b), A),
        DistSpec((m, s) -> MvLogNormal(m, to_posdef_diagonal(s)), (a, b), A),
        DistSpec(MvLogNormal, (a, alpha), A),
        DistSpec(MvLogNormal, (a,), A),
        DistSpec(s -> MvLogNormal(to_posdef_diagonal(s)), (a,), A),
        DistSpec(s -> MvLogNormal(dim, s), (alpha,), A),

        DistSpec(Dirichlet, (ones(dim),), B ./ sum(B; dims=1)),
    ]

    broken_multivariate_distributions = DistSpec[
        DistSpec(p -> Multinomial(2, p ./ sum(p)), (fill(0.5, 2),), [2 1; 0 1]),
        # Dispatch error
        DistSpec((m, A) -> MvNormalCanon(m, to_posdef(A)), (a, A), b),
        DistSpec(MvNormalCanon, (a, b), c),
        DistSpec(MvNormalCanon, (a, alpha), b),
        DistSpec(A -> MvNormalCanon(to_posdef(A)), (A,), a),
        DistSpec(MvNormalCanon, (a,), b),
        DistSpec(s -> MvNormalCanon(dim, s), (alpha,), a),
        DistSpec((m, A) -> MvNormalCanon(m, to_posdef(A)), (a, A), B),
        DistSpec(MvNormalCanon, (a, b), A),
        DistSpec(MvNormalCanon, (a, alpha), A),
        DistSpec(A -> MvNormalCanon(to_posdef(A)), (A,), B),
        DistSpec(MvNormalCanon, (a,), A),
        DistSpec(s -> MvNormalCanon(dim, s), (alpha,), A),
        # Test failure
        DistSpec((m, A) -> MvNormal(m, to_posdef(A)), (a, A), B),
        DistSpec(A -> MvNormal(to_posdef(A)), (A,), B),
        DistSpec((m, A) -> MvLogNormal(m, to_posdef(A)), (a, A), B),
        DistSpec(A -> MvLogNormal(to_posdef(A)), (A,), B),
    ]

    matrixvariate_distributions = DistSpec[
        # Matrix x
        DistSpec((n1, n2) -> MatrixBeta(dim, n1, n2), (3.0, 3.0), A, to_beta_mat),
        DistSpec((df, A) -> Wishart(df, to_posdef(A)), (3.0, A), B, to_posdef),
        DistSpec((df, A) -> InverseWishart(df, to_posdef(A)), (3.0, A), B, to_posdef),
        DistSpec((df, A) -> TuringWishart(df, to_posdef(A)), (3.0, A), B, to_posdef),
        DistSpec((df, A) -> TuringInverseWishart(df, to_posdef(A)), (3.0, A), B, to_posdef),
        DistSpec(() -> LKJ(10, 1.), (), A_big, to_corr),

        # Vector of matrices x
        DistSpec(
            (n1, n2) -> MatrixBeta(dim, n1, n2),
            (3.0, 3.0),
            [A, B],
            x -> map(to_beta_mat, x),
        ),
        DistSpec(
            (df, A) -> Wishart(df, to_posdef(A)),
            (3.0, A),
            [B, C],
            x -> map(to_posdef, x),
        ),
        DistSpec(
            (df, A) -> InverseWishart(df, to_posdef(A)),
            (3.0, A),
            [B, C],
            x -> map(to_posdef, x),
        ),
        DistSpec(
            (df, A) -> TuringWishart(df, to_posdef(A)),
            (3.0, A),
            [B, C],
            x -> map(to_posdef, x),
        ),
        DistSpec(
            (df, A) -> TuringInverseWishart(df, to_posdef(A)),
            (3.0, A),
            [B, C],
            x -> map(to_posdef, x),
        ),
        DistSpec(
            () -> LKJ(10, 1.),
            (),
            [A_big, B_big],
            x -> map(to_corr, x),
        )
    ]

    broken_matrixvariate_distributions = DistSpec[
        # Other
        DistSpec(
            (A, B, C) -> MatrixNormal(A, to_posdef(B), to_posdef(C)),
            (A, B, B),
            C,
            to_posdef,
        ),
        DistSpec(() -> MatrixNormal(dim, dim), (), A, to_posdef),
        DistSpec(
            (df, A, B, C) -> MatrixTDist(df, A, to_posdef(B), to_posdef(C)),
            (1.0, A, B, B),
            C,
            to_posdef,
        ),
        DistSpec(
            (n1, n2, A) -> MatrixFDist(n1, n2, to_posdef(A)),
            (3.0, 3.0, A),
            B,
            to_posdef,
        ),
        DistSpec((eta) -> LKJ(10, eta), (1.), A_big, to_corr) 
        # AD for parameters of LKJ requires more DistributionsAD supports
    ]

    @testset "Univariate distributions" begin
        println("\nTesting: Univariate distributions\n")

        for d in univariate_distributions
            test_ad(d)
        end
    end

    @testset "Multivariate distributions" begin
        println("\nTesting: Multivariate distributions\n")

        for d in multivariate_distributions
            test_ad(d)
        end

        # Test `filldist` and `arraydist` distributions of univariate distributions
        n = 2 # always use two distributions
        for d in univariate_distributions
            d.x isa Number || continue

            # Broken distributions
            d.f(d.θ...) isa Union{VonMises,TriangularDist} && continue

            # Create `filldist` distribution
            f_filldist = (θ...,) -> filldist(d.f(θ...), n)
            d_filldist = f_filldist(d.θ...)

            # Create `arraydist` distribution
            f_arraydist = (θ...,) -> arraydist([d.f(θ...) for _ in 1:n])
            d_arraydist = f_arraydist(d.θ...)

            for sz in ((n,), (n, 2))
                # Matrix case doesn't work for continuous distributions for some reason
                # now but not too important (?!)
                if length(sz) == 2 && Distributions.value_support(typeof(d)) === Continuous
                    continue
                end

                # Compute compatible sample
                x = fill(d.x, sz)

                # Test AD
                test_ad(
                    DistSpec(Symbol(:filldist, " (", d.name, ", $sz)"), f_filldist, d.θ, x)
                )
                test_ad(
                    DistSpec(Symbol(:arraydist, " (", d.name, ", $sz)"), f_arraydist, d.θ, x)
                )
            end
        end
    end

    @testset "Matrixvariate distributions" begin
        println("\nTesting: Matrixvariate distributions\n")

        for d in matrixvariate_distributions
            test_ad(d)
        end

        # Test `filldist` and `arraydist` distributions of univariate distributions
        n = (2, 2) # always use 2 x 2 distributions
        for d in univariate_distributions
            d.x isa Number || continue
            Distributions.value_support(typeof(d)) === Discrete && continue

            # Broken distributions
            d.f(d.θ...) isa Union{VonMises,TriangularDist} && continue

            # Create `filldist` distribution
            f_filldist = (θ...,) -> filldist(d.f(θ...), n...)

            # Create `arraydist` distribution
            f_arraydist = (θ...,) -> arraydist(fill(d.f(θ...), n...))

            # Matrix `x`
            x_mat = fill(d.x, n)

            # Test AD
            test_ad(
                DistSpec(Symbol(:filldist, " (", d.name, ", $n)"), f_filldist, d.θ, x_mat)
            )
            test_ad(
                DistSpec(Symbol(:arraydist, " (", d.name, ", $n)"), f_arraydist, d.θ, x_mat)
            )

            # Vector of matrices `x`
            x_vec_of_mat = [fill(d.x, n) for _ in 1:2]

            # Test AD
            test_ad(
                DistSpec(
                    Symbol(:filldist, " (", d.name, ", $n, 2)"),
                    f_filldist,
                    d.θ,
                    x_vec_of_mat,
                )
            )
            test_ad(
                DistSpec(
                    Symbol(:arraydist, " (", d.name, ", $n, 2)"),
                    f_arraydist,
                    d.θ,
                    x_vec_of_mat,
                )
            )
        end


        # test `filldist` and `arraydist` distributions of multivariate distributions
        n = 2 # always use two distributions
        for d in multivariate_distributions
            d.x isa AbstractVector || continue
            Distributions.value_support(typeof(d)) === Discrete && continue

            # Tests are failing for matrix covariance vectorized MvNormal
            if d.f(d.θ...) isa Union{
                MvNormal,MvLogNormal,
                DistributionsAD.TuringDenseMvNormal,
                DistributionsAD.TuringDiagMvNormal,
                DistributionsAD.TuringScalMvNormal,
                TuringMvLogNormal
            }
                any(x isa Matrix for x in d.θ) && continue
            end

            # Create `filldist` distribution
            f_filldist = (θ...,) -> filldist(d.f(θ...), n)

            # Create `arraydist` distribution
            f_arraydist = (θ...,) -> arraydist(fill(d.f(θ...), n))

            # Matrix `x`
            x_mat = repeat(d.x, 1, n)

            # Test AD
            test_ad(
                DistSpec(Symbol(:filldist, " (", d.name, ", $n)"), f_filldist, d.θ, x_mat)
            )
            test_ad(
                DistSpec(Symbol(:arraydist, " (", d.name, ", $n)"), f_arraydist, d.θ, x_mat)
            )

            # Vector of matrices `x`
            x_vec_of_mat = [repeat(d.x, 1, n) for _ in 1:2]

            # Test AD
            test_ad(
                DistSpec(
                    Symbol(:filldist, " (", d.name, ", $n, 2)"),
                    f_filldist,
                    d.θ,
                    x_vec_of_mat,
                )
            )
            test_ad(
                DistSpec(
                    Symbol(:arraydist, " (", d.name, ", $n, 2)"),
                    f_arraydist,
                    d.θ,
                    x_vec_of_mat,
                )
            )
        end
    end

    @testset "Turing issue 1385" begin
        dist = arraydist(truncated.(Laplace.(0, [1, 2]), -10.0, 70.0))
        x = ReverseDiff.track(rand(dist))
        @test typeof(bijector(dist)(x)) <: ReverseDiff.TrackedArray
    end
end
