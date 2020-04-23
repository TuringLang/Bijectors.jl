using .DistributionsAD: TuringDirichlet, TuringWishart, TuringInverseWishart,
                        FillVectorOfUnivariate, FillMatrixOfUnivariate,
                        MatrixOfUnivariate, FillVectorOfMultivariate, VectorOfMultivariate,
                        TuringScalMvNormal, TuringDiagMvNormal, TuringDenseMvNormal

# Bijectors

bijector(::TuringDirichlet) = SimplexBijector()
bijector(::TuringWishart) = PDBijector()
bijector(::TuringInverseWishart) = PDBijector()
bijector(::TuringScalMvNormal) = Identity{1}()
bijector(::TuringDiagMvNormal) = Identity{1}()
bijector(::TuringDenseMvNormal) = Identity{1}()
# Fallback on link and invlink
bijector(d::MatrixOfUnivariate) = DistributionBijector(d)
bijector(d::VectorOfMultivariate) = DistributionBijector(d)

# Non-Fill versions are not efficient for AD but are not used by Turing because we define 
# logpdf_with_trans on the distributions directly
function logabsdetjac(
    b::DistributionBijector{<:Any, <:MatrixOfUnivariate},
    x::AbstractMatrix{<:Real},
)
    return sum(logabsdetjac.(bijector.(b.dist.dists), x))
end
function logabsdetjac(
    b::DistributionBijector{<:Any, <:VectorOfMultivariate},
    x::AbstractMatrix{<:Real},
)
    return sumeachcol(x, b.dist.dists) do c, dist
        logabsdetjac(bijector(dist), c)
    end
end
function logabsdetjac(
    b::DistributionBijector{<:Any, <:FillVectorOfUnivariate},
    x::AbstractVector{<:Real},
)
    return sum(logabsdetjac(bijector(b.dist.dists.value), x))
end
function logabsdetjac(
    b::DistributionBijector{<:Any, <:FillMatrixOfUnivariate},
    x::AbstractMatrix{<:Real},
)
    return sum(logabsdetjac(bijector(b.dist.dists.value), x))
end
function logabsdetjac(
    b::DistributionBijector{<:Any, <:FillVectorOfMultivariate},
    x::AbstractMatrix{<:Real},
)
    return sum(logabsdetjac(bijector(b.dist.dists.value), x))
end

@register DistributionsAD.TuringUniform

# TuringDirichlet

function link(
    d::TuringDirichlet,
    x::AbstractVecOrMat{<:Real},
    proj::Bool = true,
)
    return SimplexBijector{proj}()(x)
end

function invlink(
    d::TuringDirichlet,
    y::AbstractVecOrMat{<:Real},
    proj::Bool = true,
)
    return inv(SimplexBijector{proj}())(y)
end

function logpdf_with_trans(
    d::TuringDirichlet,
    x::AbstractVecOrMat{<:Real},
    transform::Bool,
)
    return dirichlet_logpdf_with_trans(d, x, transform)
end

# TuringWishart

function logpdf_with_trans(
    d::TuringWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    return pd_logpdf_with_trans(d, X, transform)
end
function logpdf_with_trans(
    d::TuringWishart,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool
)
    return map(X) do x
        pd_logpdf_with_trans(d, x, transform)
    end
end
function getlogp(d::TuringWishart, Xcf, X)
    return 0.5 * ((d.df - (dim(d) + 1)) * logdet(Xcf) - tr(d.chol \ X)) - d.c0
end

# TuringInverseWishart

function logpdf_with_trans(
    d::TuringInverseWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    pd_logpdf_with_trans(d, X, transform)
end
function logpdf_with_trans(
    d::TuringInverseWishart,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool
)
    return map(X) do x
        pd_logpdf_with_trans(d, x, transform)
    end
end
function getlogp(d::TuringInverseWishart, Xcf, X)
    Ψ = d.S
    return -0.5 * ((d.df + dim(d) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end

# filldist and arraydist

function logpdf_with_trans(
    dist::FillVectorOfUnivariate{Discrete},
    x::AbstractVecOrMat{<:Real},
    istrans::Bool,
)
    return logpdf(dist, x)
end
function logpdf_with_trans(
    dist::FillVectorOfUnivariate{Continuous},
    x::AbstractVector{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans(dist.v.value, x, istrans))
end
function logpdf_with_trans(
    dist::FillVectorOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return vec(sum(logpdf_with_trans(dist.v.value, x, istrans), dims = 1))
end

link(dist::FillVectorOfUnivariate{Discrete}, x::AbstractVecOrMat{<:Real}) = copy(x)
function link(
    dist::FillVectorOfUnivariate{Continuous},
    x::AbstractVector{<:Real},
)
    return link(dist.v.value, x)
end
function link(
    dist::FillVectorOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return link(dist.v.value, x)
end

invlink(dist::FillVectorOfUnivariate{Discrete}, x::AbstractVecOrMat{<:Real}) = copy(x)
function invlink(
    dist::FillVectorOfUnivariate{Continuous},
    x::AbstractVector{<:Real},
)
    return invlink(dist.v.value, x)
end
function invlink(
    dist::FillVectorOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.v.value, x)
end

function logpdf_with_trans(
    dist::FillMatrixOfUnivariate{Discrete},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return logpdf(dist, x)
end
function logpdf_with_trans(
    dist::FillMatrixOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans(dist.dists.value, x, istrans))
end

link(dist::FillMatrixOfUnivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function link(
    dist::FillMatrixOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return link(dist.dists.value, x)
end

invlink(dist::FillMatrixOfUnivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function invlink(
    dist::FillMatrixOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.dists.value, x)
end

function logpdf_with_trans(
    dist::FillVectorOfMultivariate{Discrete},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return logpdf(dist, x)
end
function logpdf_with_trans(
    dist::FillVectorOfMultivariate{Continuous},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans(dist.dists.value, x, istrans))
end

link(dist::FillVectorOfMultivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function link(
    dist::FillVectorOfMultivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return link(dist.dists.value, x)
end

invlink(dist::FillVectorOfMultivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function invlink(
    dist::FillVectorOfMultivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.dists.value, x)
end

function logpdf_with_trans(
    dist::VectorOfMultivariate{Discrete},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return logpdf(dist, x)
end
function logpdf_with_trans(
    dist::VectorOfMultivariate{Continuous},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sumeachcol(x, dist.dists) do c, dist
        logpdf_with_trans(dist, c, istrans)
    end
end

function logpdf_with_trans(
    dist::VectorOfMultivariate{Discrete},
    x::AbstractArray{<:AbstractMatrix{<:Real}},
    istrans::Bool,
)
    return logpdf(dist, x)
end
function logpdf_with_trans(
    dist::VectorOfMultivariate{Continuous},
    x::AbstractArray{<:AbstractMatrix{<:Real}},
    istrans::Bool,
)
    return map(x) do x
        logpdf_with_trans(dist, x, istrans)
    end
end

link(dist::VectorOfMultivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function link(
    dist::VectorOfMultivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x, dist.dists) do c, dist
        link(dist, c)
    end
end
invlink(dist::VectorOfMultivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function invlink(
    dist::VectorOfMultivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x, dist.dists) do c, dist
        invlink(dist, c)
    end
end

function logpdf_with_trans(
    dist::MatrixOfUnivariate{Discrete},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return logpdf(dist, x)
end
function logpdf_with_trans(
    dist::MatrixOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(maporbroadcast(dist.dists, x) do dist, x
        logpdf_with_trans(dist, x, istrans)
    end)
end

link(dist::MatrixOfUnivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function link(
    dist::MatrixOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return maporbroadcast(link, dist.dists, x)
end
invlink(dist::MatrixOfUnivariate{Discrete}, x::AbstractMatrix{<:Real}) = copy(x)
function invlink(
    dist::MatrixOfUnivariate{Continuous},
    x::AbstractMatrix{<:Real},
)
    return maporbroadcast(invlink, dist.dists, x)
end

function logpdf_with_trans(
    dist::Union{MatrixOfUnivariate, VectorOfMultivariate},
    x::AbstractArray{<:AbstractMatrix{<:Real}},
    istrans::Bool,
)
    map(x) do x
        logpdf_with_trans(dist, x, istrans)
    end
end
function link(
    dist::Union{MatrixOfUnivariate, VectorOfMultivariate},
    x::AbstractArray{<:AbstractMatrix{<:Real}},
)
    map(x) do x
        link(dist, x)
    end
end
function invlink(
    dist::Union{MatrixOfUnivariate, VectorOfMultivariate},
    x::AbstractArray{<:AbstractMatrix{<:Real}},
)
    map(x) do x
        invlink(dist, x)
    end
end