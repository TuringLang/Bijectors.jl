macro register(dist)
    return quote
        Bijectors.eval(getexpr($(esc(dist))))
        toflatten(::$(esc(dist))) = true
    end
end
function getexpr(Tdist)
    x = gensym(:x)
    fnames = fieldnames(Tdist)
    flattened_args = Expr(:tuple, [:(dist.$f) for f in fnames]...)
    func = Expr(:->, 
                Expr(:tuple, fnames..., x), 
                Expr(:block,
                    Expr(:call, :logpdf_with_trans,
                        Expr(:call, :($Tdist), fnames...),
                        x,
                        :trans,
                    )
                )
            )
    return :(flatten(dist::$Tdist, trans) = ($func, $flattened_args))
end
const flattened_dists = [   Bernoulli,
                            BetaBinomial,
                            Binomial,
                            Geometric,
                            NegativeBinomial,
                            Poisson,
                            Skellam,
                            Arcsine,
                            Beta,
                            BetaPrime,
                            Biweight,
                            Cauchy,
                            Chernoff,
                            Chi,
                            Chisq,
                            Cosine,
                            Epanechnikov,
                            Erlang,
                            Exponential,
                            FDist,
                            Frechet,
                            Gamma,
                            GeneralizedExtremeValue,
                            GeneralizedPareto,
                            Gumbel,
                            #InverseGamma,
                            InverseGaussian,
                            Kolmogorov,
                            Laplace,
                            Levy,
                            LocationScale,
                            Logistic,
                            LogitNormal,
                            LogNormal,
                            Normal,
                            #NormalCanon,
                            #NormalInverseGaussian,
                            Pareto,
                            PGeneralizedGaussian,
                            Rayleigh,
                            SymTriangularDist,
                            TDist,
                            TriangularDist,
                            Triweight,
                            #Truncated,
                            #VonMises,
                        ]
for T in flattened_dists
    @eval toflatten(::$T) = true
end
toflatten(::Distribution) = false
for T in flattened_dists
    eval(getexpr(T))
end
