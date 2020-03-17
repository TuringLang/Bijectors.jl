@testset "AD tests" begin
    ddim = 3
    dmean = zeros(ddim)
    cov_mat = Matrix{Float64}(I, ddim, ddim)
    cov_vec = ones(ddim)
    cov_num = 1.0
    norm_val_vec = ones(ddim)
    norm_val_mat = ones(ddim, 2)
    alpha = ones(4)
    dir_val_vec = fill(0.25, 4)
    dir_val_mat = fill(0.25, 4, 2)
    beta_mat = rand(MatrixBeta(ddim, ddim, ddim))
    tested = []
    function test_info(name)
        if !(name in tested)
            push!(tested, name)
            @info("Testing: $(name)")
        end
    end

    function filldist_expr(sym, n = 1; dims = (1,))
        x = [gensym(:x) for i in 1:n]
        :(($(x...),) -> filldist($sym($(x...),), $(dims...)))
    end
    function arraydist_expr(sym, n = 1; dims = (1,))
        x = [gensym(:x) for i in 1:n]
        :(($(x...),) -> arraydist(reshape(fill($sym($(x...),), prod($(dims))), $(dims...))))
    end
    function filldist_spec(dist::DistSpec; disttype = :uni, n = 2, d = 1)
        @assert disttype in (:uni, :multi)
        disttype == :uni && dist.x isa Vector && return
        disttype == :multi && dist.x isa Matrix && return
        # Broken
        (dist.name in (:VonMises, :TriangularDist)) && return
        (dist.name == :Weibull && n isa Tuple) && return
        # Tests are failing for matrix covariance vectorized MvNormal
        # Note the 2 MvNormal cases in broken_multi_cont_dists
        if dist.name in (:MvNormal, :TuringMvNormal, :MvLogNormal, :TuringMvNormal) && 
            any(x -> isa(x, Matrix), dist.θ)
            return
        end
        if disttype == :multi
            name = filldist_expr(dist.name, length(dist.θ); dims = (n,))
            x = hcat(fill(dist.x, n)...,)
            x = d == 1 ? x : fill(x, d)
        else
            if n isa Tuple
                name = filldist_expr(dist.name, length(dist.θ); dims = n)
                x = fill(dist.x, n...,)
                x = d == 1 ? x : fill(x, d)
            else # n number
                name = filldist_expr(dist.name, length(dist.θ); dims = (n,))
                x = fill(dist.x, n)
                x = d == 1 ? x : hcat(fill(x, d)...,)
            end
        end
        return DistSpec(name, dist.θ, x)
    end
    function arraydist_spec(dist::DistSpec; disttype = :uni, n = 2, d = 1)
        @assert disttype in (:uni, :multi)
        (disttype == :uni) && (dist.x isa Vector) && return
        (disttype == :multi) && (dist.x isa Matrix) && return
        # Broken
        (dist.name in (:VonMises, :TriangularDist)) && return
        (dist.name == :NormalInverseGaussian && n isa Tuple) && return
        if dist.name in (:MvNormal, :TuringMvNormal, :MvLogNormal, :TuringMvNormal) && 
            any(x -> isa(x, Matrix), dist.θ)
            return
        end
        if disttype == :multi
            name = arraydist_expr(dist.name, length(dist.θ); dims = (n,))
            x = hcat(fill(dist.x, n)...,)
            x = d == 1 ? x : fill(x, d)
        else
            if n isa Tuple
                name = arraydist_expr(dist.name, length(dist.θ); dims = n)
                x = fill(dist.x, n...,)
                x = d == 1 ? x : fill(x, d)
            else
                name = arraydist_expr(dist.name, length(dist.θ); dims = (n,))
                x = fill(dist.x, n)
                x = d == 1 ? x : hcat(fill(x, d)...,)
            end
        end
        return DistSpec(name, dist.θ, x)
    end

    uni_disc_dists = [
        DistSpec(:Bernoulli, (0.45,), 1),
        DistSpec(:Bernoulli, (0.45,), [1, 1]),
        DistSpec(:Bernoulli, (0.45,), 0),
        DistSpec(:Bernoulli, (0.45,), [0, 0]),
    
        DistSpec(:((a, b) -> BetaBinomial(10, a, b)), (2.0, 1.0), 5),
        DistSpec(:((a, b) -> BetaBinomial(10, a, b)), (2.0, 1.0), [5, 5]),
    
        DistSpec(:(p -> Binomial(10, p)), (0.5,), 5),
        DistSpec(:(p -> Binomial(10, p)), (0.5,), [5, 5]),
    
        DistSpec(:(p -> Categorical(p / sum(p))), ([0.45, 0.55],), 1),
        DistSpec(:(p -> Categorical(p / sum(p))), ([0.45, 0.55],), [1, 1]),
    
        DistSpec(:Geometric, (0.45,), 3),
        DistSpec(:Geometric, (0.45,), [3, 3]),
    
        DistSpec(:NegativeBinomial, (3.5, 0.5), 1),
        DistSpec(:NegativeBinomial, (3.5, 0.5), [1, 1]),
    
        DistSpec(:Poisson, (0.5,), 1),
        DistSpec(:Poisson, (0.5,), [1, 1]),
    
        DistSpec(:Skellam, (1.0, 2.0), -2),
        DistSpec(:Skellam, (1.0, 2.0), [-2, -2]),
    
        DistSpec(:PoissonBinomial, ([0.5, 0.5],), 0),
        DistSpec(:PoissonBinomial, ([0.5, 0.5],), [0, 0]),
    
        DistSpec(:TuringPoissonBinomial, ([0.5, 0.5],), 0),
        DistSpec(:TuringPoissonBinomial, ([0.5, 0.5],), [0, 0]),
    ]
    
    uni_cont_dists = [
        DistSpec(:Arcsine, (), 0.5),
        DistSpec(:Arcsine, (1.0,), 0.5),
        DistSpec(:Arcsine, (0.0, 2.0), 0.5),
        DistSpec(:Arcsine, (), [0.5]),
        #DistSpec(:Arcsine, (1.0,), [0.5]),
        #DistSpec(:Arcsine, (0.0, 2.0), [0.5]),
    
        DistSpec(:Beta, (), 0.5),
        DistSpec(:Beta, (1.0,), 0.5),
        DistSpec(:Beta, (1.0, 2.0), 0.5),
        #DistSpec(:Beta, (), [0.5]),
        #DistSpec(:Beta, (1.0,), [0.5]),
        #DistSpec(:Beta, (1.0, 2.0), [0.5]),
    
        DistSpec(:BetaPrime, (), 0.5),
        DistSpec(:BetaPrime, (1.0,), 0.5),
        DistSpec(:BetaPrime, (1.0, 2.0), 0.5),
        #DistSpec(:BetaPrime, (), [0.5]),
        #DistSpec(:BetaPrime, (1.0,), [0.5]),
        #DistSpec(:BetaPrime, (1.0, 2.0), [0.5]),
    
        DistSpec(:Biweight, (), 0.5),
        DistSpec(:Biweight, (1.0,), 0.5),
        DistSpec(:Biweight, (1.0, 2.0), 0.5),
        #DistSpec(:Biweight, (), [0.5]),
        #DistSpec(:Biweight, (1.0,), [0.5]),
        #DistSpec(:Biweight, (1.0, 2.0), [0.5]),
    
        DistSpec(:Cauchy, (), 0.5),
        DistSpec(:Cauchy, (1.0,), 0.5),
        DistSpec(:Cauchy, (1.0, 2.0), 0.5),
        #DistSpec(:Cauchy, (), [0.5]),
        #DistSpec(:Cauchy, (1.0,), [0.5]),
        #DistSpec(:Cauchy, (1.0, 2.0), [0.5]),
    
        DistSpec(:Chi, (1.0,), 0.5),
        #DistSpec(:Chi, (1.0,), [0.5]),
    
        DistSpec(:Chisq, (1.0,), 0.5),
        #DistSpec(:Chisq, (1.0,), [0.5]),
    
        DistSpec(:Cosine, (1.0, 1.0), 0.5),
        #DistSpec(:Cosine, (1.0, 1.0), [0.5]),
    
        DistSpec(:Epanechnikov, (1.0, 1.0), 0.5),
        #DistSpec(:Epanechnikov, (1.0, 1.0), [0.5]),
    
        DistSpec(:((s)->Erlang(1, s)), (1.0,), 0.5), # First arg is integer
        #DistSpec(:((s)->Erlang(1, s)), (1.0,), [0.5]),
    
        DistSpec(:Exponential, (1.0,), 0.5),
        #DistSpec(:Exponential, (1.0,), [0.5]),
    
        DistSpec(:FDist, (1.0, 1.0), 0.5),
        #DistSpec(:FDist, (1.0, 1.0), [0.5]),
    
        DistSpec(:Frechet, (), 0.5),
        DistSpec(:Frechet, (1.0,), 0.5),
        DistSpec(:Frechet, (1.0, 2.0), 0.5),
        #DistSpec(:Frechet, (), [0.5]),
        #DistSpec(:Frechet, (1.0,), [0.5]),
        #DistSpec(:Frechet, (1.0, 2.0), [0.5]),
    
        DistSpec(:Gamma, (), 0.5),
        DistSpec(:Gamma, (1.0,), 0.5),
        DistSpec(:Gamma, (1.0, 2.0), 0.5),
        #DistSpec(:Gamma, (), [0.5]),
        #DistSpec(:Gamma, (1.0,), [0.5]),
        #DistSpec(:Gamma, (1.0, 2.0), [0.5]),
    
        DistSpec(:GeneralizedExtremeValue, (1.0, 1.0, 1.0), 0.5),
        #DistSpec(:GeneralizedExtremeValue, (1.0, 1.0, 1.0), [0.5]),
    
        DistSpec(:GeneralizedPareto, (), 0.5),
        DistSpec(:GeneralizedPareto, (1.0, 2.0), 0.5),
        DistSpec(:GeneralizedPareto, (0.0, 2.0, 3.0), 0.5),
        #DistSpec(:GeneralizedPareto, (), [0.5]),
        #DistSpec(:GeneralizedPareto, (1.0, 2.0), [0.5]),
        #DistSpec(:GeneralizedPareto, (0.0, 2.0, 3.0), [0.5]),
    
        DistSpec(:Gumbel, (), 0.5),
        DistSpec(:Gumbel, (1.0,), 0.5),
        DistSpec(:Gumbel, (1.0, 2.0), 0.5),
        #DistSpec(:Gumbel, (), [0.5]),
        #DistSpec(:Gumbel, (1.0,), [0.5]),
        #DistSpec(:Gumbel, (1.0, 2.0), [0.5]),
    
        DistSpec(:InverseGamma, (), 0.5),
        DistSpec(:InverseGamma, (1.0,), 0.5),
        DistSpec(:InverseGamma, (1.0, 2.0), 0.5),
        #DistSpec(:InverseGamma, (), [0.5]),
        #DistSpec(:InverseGamma, (1.0,), [0.5]),
        #DistSpec(:InverseGamma, (1.0, 2.0), [0.5]),
    
        DistSpec(:InverseGaussian, (), 0.5),
        DistSpec(:InverseGaussian, (1.0,), 0.5),
        DistSpec(:InverseGaussian, (1.0, 2.0), 0.5),
        #DistSpec(:InverseGaussian, (), [0.5]),
        #DistSpec(:InverseGaussian, (1.0,), [0.5]),
        #DistSpec(:InverseGaussian, (1.0, 2.0), [0.5]),
    
        DistSpec(:Kolmogorov, (), 0.5),
        #DistSpec(:Kolmogorov, (), [0.5]),
    
        DistSpec(:Laplace, (), 0.5),
        DistSpec(:Laplace, (1.0,), 0.5),
        DistSpec(:Laplace, (1.0, 2.0), 0.5),
        #DistSpec(:Laplace, (), [0.5]),
        #DistSpec(:Laplace, (1.0,), [0.5]),
        #DistSpec(:Laplace, (1.0, 2.0), [0.5]),
    
        DistSpec(:Levy, (), 0.5),
        DistSpec(:Levy, (0.0,), 0.5),
        DistSpec(:Levy, (0.0, 2.0), 0.5),
        #DistSpec(:Levy, (), [0.5]),
        #DistSpec(:Levy, (0.0,), [0.5]),
        #DistSpec(:Levy, (0.0, 2.0), [0.5]),
    
        DistSpec(:((a, b) -> LocationScale(a, b, Normal())), (1.0, 2.0), 0.5),
        #DistSpec(:((a, b) -> LocationScale(a, b, Normal())), (1.0, 2.0), [0.5]),
    
        DistSpec(:Logistic, (), 0.5),
        DistSpec(:Logistic, (1.0,), 0.5),
        DistSpec(:Logistic, (1.0, 2.0), 0.5),
        #DistSpec(:Logistic, (), [0.5]),
        #DistSpec(:Logistic, (1.0,), [0.5]),
        #DistSpec(:Logistic, (1.0, 2.0), [0.5]),
    
        DistSpec(:LogitNormal, (), 0.5),
        DistSpec(:LogitNormal, (1.0,), 0.5),
        DistSpec(:LogitNormal, (1.0, 2.0), 0.5),
        #DistSpec(:LogitNormal, (), [0.5]),
        #DistSpec(:LogitNormal, (1.0,), [0.5]),
        #DistSpec(:LogitNormal, (1.0, 2.0), [0.5]),
    
        DistSpec(:LogNormal, (), 0.5),
        DistSpec(:LogNormal, (1.0,), 0.5),
        DistSpec(:LogNormal, (1.0, 2.0), 0.5),
        #DistSpec(:LogNormal, (), [0.5]),
        #DistSpec(:LogNormal, (1.0,), [0.5]),
        #DistSpec(:LogNormal, (1.0, 2.0), [0.5]),
    
        DistSpec(:Normal, (), 0.5),
        DistSpec(:Normal, (1.0,), 0.5),
        DistSpec(:Normal, (1.0, 2.0), 0.5),
        #DistSpec(:Normal, (), [0.5]),
        #DistSpec(:Normal, (1.0,), [0.5]),
        #DistSpec(:Normal, (1.0, 2.0), [0.5]),
    
        DistSpec(:NormalCanon, (1.0, 2.0), 0.5),
        #DistSpec(:NormalCanon, (1.0, 2.0), [0.5]),
    
        DistSpec(:NormalInverseGaussian, (1.0, 2.0, 1.0, 1.0), 0.5),
        #DistSpec(:NormalInverseGaussian, (1.0, 2.0, 1.0, 1.0), [0.5]),
        
        DistSpec(:Pareto, (), 1.5),
        DistSpec(:Pareto, (1.0,), 1.5),
        DistSpec(:Pareto, (1.0, 1.0), 1.5),
        #DistSpec(:Pareto, (), [1.5]),
        #DistSpec(:Pareto, (1.0,), [1.5]),
        #DistSpec(:Pareto, (1.0, 1.0), [1.5]),
    
        DistSpec(:PGeneralizedGaussian, (), 0.5),
        DistSpec(:PGeneralizedGaussian, (1.0, 1.0, 1.0), 0.5),
        #DistSpec(:PGeneralizedGaussian, (), [0.5]),
        #DistSpec(:PGeneralizedGaussian, (1.0, 1.0, 1.0), [0.5]),
    
        DistSpec(:Rayleigh, (), 0.5),
        DistSpec(:Rayleigh, (1.0,), 0.5),
        #DistSpec(:Rayleigh, (), [0.5]),
        #DistSpec(:Rayleigh, (1.0,), [0.5]),
    
        DistSpec(:Semicircle, (1.0,), 0.5),
        #DistSpec(:Semicircle, (1.0,), [0.5]),
    
        DistSpec(:SymTriangularDist, (), 0.5),
        DistSpec(:SymTriangularDist, (1.0,), 0.5),
        DistSpec(:SymTriangularDist, (1.0, 2.0), 0.5),
        #DistSpec(:SymTriangularDist, (), [0.5]),
        #DistSpec(:SymTriangularDist, (1.0,), [0.5]),
        #DistSpec(:SymTriangularDist, (1.0, 2.0), [0.5]),
    
        DistSpec(:TDist, (1.0,), 0.5),
        #DistSpec(:TDist, (1.0,), [0.5]),
    
        DistSpec(:TriangularDist, (1.0, 2.0), 1.5),
        DistSpec(:TriangularDist, (1.0, 3.0, 2.0), 1.5),
        #DistSpec(:TriangularDist, (1.0, 2.0), [1.5]),
        #DistSpec(:TriangularDist, (1.0, 3.0, 2.0), [1.5]),
    
        DistSpec(:Triweight, (1.0, 1.0), 1.0),
        #DistSpec(:Triweight, (1.0, 1.0), [1.0]),
    
        DistSpec(:((mu, sigma, l, u) -> truncated(Normal(mu, sigma), l, u)), (0.0, 1.0, 1.0, 2.0), 1.5),
        #DistSpec(:((mu, sigma, l, u) -> truncated(Normal(mu, sigma), l, u)), (0.0, 1.0, 1.0, 2.0), [1.5]),
    
        DistSpec(:Uniform, (), 0.5),
        DistSpec(:Uniform, (0.0, 1.0), 0.5),
        #DistSpec(:Uniform, (), [0.5]),
        #DistSpec(:Uniform, (0.0, 1.0), [0.5]),
    
        DistSpec(:TuringUniform, (), 0.5),
        DistSpec(:TuringUniform, (0.0, 1.0), 0.5),
        #DistSpec(:TuringUniform, (), [0.5]),
        #DistSpec(:TuringUniform, (0.0, 1.0), [0.5]),
    
        DistSpec(:VonMises, (), 1.0),
        #DistSpec(:VonMises, (), [1.0]),

        DistSpec(:Weibull, (), 1.0),
        DistSpec(:Weibull, (1.0,), 1.0),
        DistSpec(:Weibull, (1.0, 1.0), 1.0),
        #DistSpec(:Weibull, (), [1.0]),
        #DistSpec(:Weibull, (1.0,), [1.0]),
        #DistSpec(:Weibull, (1.0, 1.0), [1.0]),
    ]
    broken_uni_cont_dists = [
        # Zygote
        DistSpec(:Chernoff, (), 0.5),
        # Broken in Distributions even without autodiff
        DistSpec(:(()->KSDist(1)), (), 0.5), 
        DistSpec(:(()->KSOneSided(1)), (), 0.5), 
        DistSpec(:StudentizedRange, (1.0, 2.0), 0.5),
        # Dispatch error caused by ccall
        DistSpec(:NoncentralBeta, (1.0, 2.0, 1.0), 0.5), 
        DistSpec(:NoncentralChisq, (1.0, 2.0), 0.5),
        DistSpec(:NoncentralF, (1, 2, 1), 0.5),
        DistSpec(:NoncentralT, (1, 2), 0.5),
        # Stackoverflow caused by SpecialFunctions.besselix
        DistSpec(:VonMises, (1.0,), 1.0),
        DistSpec(:VonMises, (1, 1), 1),
    ]
    
    multi_disc_dists = [
        # Vector x
        DistSpec(:((p) -> Multinomial(2, p / sum(p))), (fill(0.5, 2),), [2, 0]),
        # Matrix x
        #DistSpec(:((p) -> Multinomial(2, p / sum(p))), (fill(0.5, 2),), [2 2; 0 0]);
    ]
    xmulti_disc_dists = [
        multi_disc_dists;
        # Vector x
        filter(!isnothing, filldist_spec.(uni_disc_dists; n = 2, d = 1));
        filter(!isnothing, arraydist_spec.(uni_disc_dists; n = 2, d = 1));
        # Matrix x
        filter(!isnothing, filldist_spec.(uni_disc_dists; n = 2, d = 2));
        filter(!isnothing, arraydist_spec.(uni_disc_dists; n = 2, d = 2));
    ]
    
    multi_cont_dists = [
        # Vector x
        DistSpec(:MvNormal, (dmean, cov_mat), norm_val_vec),
        DistSpec(:MvNormal, (dmean, cov_vec), norm_val_vec),
        DistSpec(:MvNormal, (dmean, Diagonal(cov_vec)), norm_val_vec),
        DistSpec(:MvNormal, (dmean, cov_num), norm_val_vec),
        DistSpec(:((m, v) -> MvNormal(m, v*I)), (dmean, cov_num), norm_val_vec),
        DistSpec(:MvNormal, (cov_mat,), norm_val_vec),
        DistSpec(:MvNormal, (cov_vec,), norm_val_vec),
        DistSpec(:MvNormal, (Diagonal(cov_vec),), norm_val_vec),
        DistSpec(:(cov_num -> MvNormal($ddim, cov_num)), (cov_num,), norm_val_vec),
        DistSpec(:TuringMvNormal, (dmean, cov_mat), norm_val_vec),
        DistSpec(:TuringMvNormal, (dmean, cov_vec), norm_val_vec),
        DistSpec(:TuringMvNormal, (dmean, Diagonal(cov_vec)), norm_val_vec),
        DistSpec(:TuringMvNormal, (dmean, cov_num), norm_val_vec),
        DistSpec(:((m, v) -> TuringMvNormal(m, v*I)), (dmean, cov_num), norm_val_vec),
        DistSpec(:TuringMvNormal, (cov_mat,), norm_val_vec),
        DistSpec(:TuringMvNormal, (cov_vec,), norm_val_vec),
        DistSpec(:TuringMvNormal, (Diagonal(cov_vec),), norm_val_vec),
        DistSpec(:(cov_num -> TuringMvNormal($ddim, cov_num)), (cov_num,), norm_val_vec),
        DistSpec(:MvLogNormal, (dmean, cov_mat), norm_val_vec),
        DistSpec(:MvLogNormal, (dmean, cov_vec), norm_val_vec),
        DistSpec(:MvLogNormal, (dmean, Diagonal(cov_vec)), norm_val_vec),
        DistSpec(:MvLogNormal, (dmean, cov_num), norm_val_vec),
        DistSpec(:MvLogNormal, (cov_mat,), norm_val_vec),
        DistSpec(:MvLogNormal, (cov_vec,), norm_val_vec),
        DistSpec(:MvLogNormal, (Diagonal(cov_vec),), norm_val_vec),
        DistSpec(:(cov_num -> MvLogNormal($ddim, cov_num)), (cov_num,), norm_val_vec),

        DistSpec(:Dirichlet, (alpha,), dir_val_vec),

        # Matrix case
        DistSpec(:MvNormal, (dmean, cov_vec), norm_val_mat),
        DistSpec(:MvNormal, (dmean, Diagonal(cov_vec)), norm_val_mat),
        DistSpec(:MvNormal, (dmean, cov_num), norm_val_mat),
        DistSpec(:((m, v) -> MvNormal(m, v*I)), (dmean, cov_num), norm_val_mat),
        DistSpec(:MvNormal, (cov_vec,), norm_val_mat),
        DistSpec(:MvNormal, (Diagonal(cov_vec),), norm_val_mat),
        DistSpec(:(cov_num -> MvNormal($ddim, cov_num)), (cov_num,), norm_val_mat),
        DistSpec(:MvLogNormal, (dmean, cov_vec), norm_val_mat),
        DistSpec(:MvLogNormal, (dmean, Diagonal(cov_vec)), norm_val_mat),
        DistSpec(:MvLogNormal, (dmean, cov_num), norm_val_mat),
        DistSpec(:MvLogNormal, (cov_vec,), norm_val_mat),
        DistSpec(:MvLogNormal, (Diagonal(cov_vec),), norm_val_mat),
        DistSpec(:(cov_num -> MvLogNormal($ddim, cov_num)), (cov_num,), norm_val_mat),

        DistSpec(:Dirichlet, (alpha,), dir_val_mat),
    ]
    xmulti_cont_dists = [
        # Vector x
        filter(!isnothing, filldist_spec.(uni_cont_dists; n = 2, d = 1));
        filter(!isnothing, arraydist_spec.(uni_cont_dists; n = 2, d = 1));

        # Matrix case
        # Doesn't work for some reason now but not too important
        #filter(!isnothing, filldist_spec.(uni_cont_dists; n = 2, d = 2));
        #filter(!isnothing, arraydist_spec.(uni_cont_dists; n = 2, d = 2));

        multi_cont_dists;
    ]
    broken_multi_cont_dists = [
        # Dispatch error
        DistSpec(:MvNormalCanon, (dmean, cov_mat), norm_val_vec),
        DistSpec(:MvNormalCanon, (dmean, cov_vec), norm_val_vec),
        DistSpec(:MvNormalCanon, (dmean, cov_num), norm_val_vec),
        DistSpec(:MvNormalCanon, (cov_mat,), norm_val_vec),
        DistSpec(:MvNormalCanon, (cov_vec,), norm_val_vec),
        DistSpec(:(cov_num -> MvNormalCanon($ddim, cov_num)), (cov_num,), norm_val_vec),
        DistSpec(:MvNormalCanon, (dmean, cov_mat), norm_val_mat),
        DistSpec(:MvNormalCanon, (dmean, cov_vec), norm_val_mat),
        DistSpec(:MvNormalCanon, (dmean, cov_num), norm_val_mat),
        DistSpec(:MvNormalCanon, (cov_mat,), norm_val_mat),
        DistSpec(:MvNormalCanon, (cov_vec,), norm_val_mat),
        DistSpec(:(cov_num -> MvNormalCanon($ddim, cov_num)), (cov_num,), norm_val_mat),
        # Test failure
        DistSpec(:MvNormal, (dmean, cov_mat), norm_val_mat),
        DistSpec(:MvNormal, (cov_mat,), norm_val_mat),
        DistSpec(:MvLogNormal, (dmean, cov_mat), norm_val_mat),
        DistSpec(:MvLogNormal, (cov_mat,), norm_val_mat),
    ]
    
    matrix_cont_dists = [
        # Matrix x
        DistSpec(:((n1, n2)->MatrixBeta($ddim, n1, n2)), (3.0, 3.0), beta_mat),
        DistSpec(:Wishart, (3.0, cov_mat), cov_mat),
        DistSpec(:InverseWishart, (3.0, cov_mat), cov_mat),
        DistSpec(:TuringWishart, (3.0, cov_mat), cov_mat),
        DistSpec(:TuringInverseWishart, (3.0, cov_mat), cov_mat),
        # Vector of matrices x
        DistSpec(:((n1, n2)->MatrixBeta($ddim, n1, n2)), (3.0, 3.0), fill(beta_mat, 2)),
        DistSpec(:Wishart, (3.0, cov_mat), fill(cov_mat, 2)),
        DistSpec(:InverseWishart, (3.0, cov_mat), fill(cov_mat, 2)),
        DistSpec(:TuringWishart, (3.0, cov_mat), fill(cov_mat, 2)),
        DistSpec(:TuringInverseWishart, (3.0, cov_mat), fill(cov_mat, 2)),
    ]
    xmatrix_cont_dists = [
        # Matrix x
        filter(!isnothing, filldist_spec.(uni_cont_dists; n = (2, 2)));
        filter(!isnothing, filldist_spec.(multi_cont_dists; disttype = :multi, n = 2));
        filter(!isnothing, arraydist_spec.(uni_cont_dists; n = (2, 2)));
        filter(!isnothing, arraydist_spec.(multi_cont_dists; disttype = :multi, n = 2));

        # Vector of matrices x
        filter(!isnothing, filldist_spec.(uni_cont_dists; n = (2, 2), d = 2));    
        filter(!isnothing, filldist_spec.(multi_cont_dists; disttype = :multi, n = 2, d = 2));
        filter(!isnothing, arraydist_spec.(uni_cont_dists; n = (2, 2), d = 2));    
        filter(!isnothing, arraydist_spec.(multi_cont_dists; disttype = :multi, n = 2, d = 2));

        matrix_cont_dists;
    ]
    broken_matrix_cont_dists = [
        # Other
        DistSpec(:MatrixNormal, (cov_mat, cov_mat, cov_mat), cov_mat),
        DistSpec(:(()->MatrixNormal($ddim, $ddim)), (), cov_mat),
        DistSpec(:MatrixTDist, (1.0, cov_mat, cov_mat, cov_mat), cov_mat),
        DistSpec(:MatrixFDist, (3.0, 3.0, cov_mat), cov_mat),
    ]

    test_head(s) = println("\n"*s*"\n")
    separator() = println("\n"*"="^50)

    separator()
    @testset "Univariate discrete distributions" begin
        test_head("Testing: Univariate discrete distributions")
        for d in uni_disc_dists
            test_info(d.name)
            for testf in get_all_functions(d, false)
                test_ad(testf.f, testf.x)
            end
        end
    end
    separator()

    # Note: broadcasting logpdf with univariate distributions having tracked parameters breaks
    # Tracker. Ref: https://github.com/FluxML/Tracker.jl/issues/65
    # filldist works around this so it is the recommended way for AD-friendly "broadcasting"
    # of logpdf with univariate distributions

    @testset "Univariate continuous distributions" begin
        test_head("Testing: Univariate continuous distributions")
        for d in uni_cont_dists
            test_info(d.name)
            for testf in get_all_functions(d, true)
                test_ad(testf.f, testf.x)
            end
        end
    end
    separator()

    @testset "Multivariate discrete distributions" begin
        test_head("Testing: Multivariate discrete distributions")
        for d in xmulti_disc_dists
            test_info(d.name)
            for testf in get_all_functions(d, false)
                test_ad(testf.f, testf.x)
            end
        end
    end

    separator()
    @testset "Multivariate continuous distributions" begin
        test_head("Testing: Multivariate continuous distributions")
        for d in xmulti_cont_dists
            test_info(d.name)
            for testf in get_all_functions(d, true)
                test_ad(testf.f, testf.x)
            end
        end
    end
    separator()

    @testset "Matrix-variate continuous distributions" begin
        test_head("Testing: Matrix-variate continuous distributions")
        for d in xmatrix_cont_dists
            test_info(d.name)
            for testf in get_all_functions(d, true)
                test_ad(testf.f, testf.x)
            end
        end
    end
    separator()
end
