using .DistributionsAD: TuringDirichlet, TuringWishart, TuringInverseWishart,
                        FillVectorOfUnivariate, FillMatrixOfUnivariate, VectorOfUnivariate,
                        MatrixOfUnivariate, FillVectorOfMultivariate, VectorOfMultivariate

# TuringDirichlet

_clamp(x, ::TuringDirichlet) = _clamp(x, SimplexBijector())
bijector(d::TuringDirichlet) = SimplexBijector()

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
    ϵ = _eps(eltype(x))
    lp = logpdf(d, x .+ ϵ)
    if transform
        lp -= logabsdetjac(bijector(d), x)
    end
    return lp
end

# TuringWishart

function logpdf_with_trans(
    d::TuringWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    return _logpdf_with_trans_pd(d, X, transform)
end
function logpdf_with_trans(
    d::TuringWishart,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool
)
    return mapvcat(X) do x
        _logpdf_with_trans_pd(d, x, transform)
    end
end
link(d::TuringWishart, X::AbstractMatrix{<:Real}) = PDBijector()(X)
invlink(d::TuringWishart, Y::AbstractMatrix{<:Real}) = inv(PDBijector())(Y)
function getlogp(d::TuringWishart, Xcf, X)
    return 0.5 * ((d.df - (dim(d) + 1)) * logdet(Xcf) - tr(d.chol \ X)) - d.c0
end

# TuringInverseWishart

function logpdf_with_trans(
    d::TuringInverseWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    _logpdf_with_trans_pd(d, X, transform)
end
function logpdf_with_trans(
    d::TuringInverseWishart,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool
)
    return mapvcat(X) do x
        _logpdf_with_trans_pd(d, x, transform)
    end
end
link(d::TuringInverseWishart, X::AbstractMatrix{<:Real}) = PDBijector()(X)
invlink(d::TuringInverseWishart, Y::AbstractMatrix{<:Real}) = inv(PDBijector())(Y)
function getlogp(d::TuringInverseWishart, Xcf, X)
    Ψ = d.S
    return -0.5 * ((d.df + dim(d) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end

# filldist and arraydist

function logpdf_with_trans(
    dist::FillVectorOfUnivariate,
    x::AbstractVector{<:Real},
    istrans::Bool,
)
    return _sum(x) do x
        logpdf_with_trans(dist.v.value, x, istrans)
    end
end
function logpdf_with_trans(
    dist::FillVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return vec(sum(logpdf_with_trans(dist.v.value, x, istrans), dims = 1))
end
function link(
    dist::FillVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return link(dist.v.value, x)
end
function link(
    dist::FillVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return link(dist.v.value, x)
end
function invlink(
    dist::FillVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return invlink(dist.v.value, x)
end
function invlink(
    dist::FillVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.v.value, x)
end

function logpdf_with_trans(
    dist::FillMatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return _sum(x) do x
        logpdf_with_trans(dist.dists.value, x, istrans)
    end
end
function logpdf_with_trans(
    dist::FillMatrixOfUnivariate,
    x::AbstractArray{<:AbstractMatrix{<:Real}},
    istrans::Bool,
)
    return mapvcat(x) do x
        logpdf_with_trans(dist, x, istrans)
    end
end
function link(
    dist::FillMatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return link(dist.dists.value, x)
end
function invlink(
    dist::FillMatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.dists.value, x)
end

function logpdf_with_trans(
    dist::FillVectorOfMultivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans(dist.dists.value, x, istrans))
end
function logpdf_with_trans(
    dist::FillVectorOfMultivariate,
    x::AbstractArray{<:AbstractMatrix{<:Real}},
    istrans::Bool,
)
    return mapvcat(x) do x
        logpdf_with_trans(dist, x, istrans)
    end
end
function link(
    dist::FillVectorOfMultivariate,
    x::AbstractMatrix{<:Real},
)
    return link(dist.dists.value, x)
end
function invlink(
    dist::FillVectorOfMultivariate,
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.dists.value, x)
end

function logpdf_with_trans(
    dist::VectorOfMultivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return _sumeachcol(x, dist.dists) do c, dist
        logpdf_with_trans(dist, c, istrans)
    end
end
function logpdf_with_trans(
    dist::VectorOfMultivariate,
    x::AbstractArray{<:AbstractMatrix{<:Real}},
    istrans::Bool,
)
    return mapvcat(x) do x
        logpdf_with_trans(dist, x, istrans)
    end
end
function link(
    dist::VectorOfMultivariate,
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x, dist.dists) do c, dist
        link(dist, c)
    end
end
function invlink(
    dist::VectorOfMultivariate,
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x, dist.dists) do c, dist
        invlink(dist, c)
    end
end

function logpdf_with_trans(
    dist::VectorOfUnivariate,
    x::AbstractVector{<:Real},
    istrans::Bool,
)
    return _sum(dist.v, x) do dist, x
        logpdf_with_trans(dist, x, istrans)
    end
end
function logpdf_with_trans(
    dist::VectorOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return mapvcat(1:size(x,2)) do i
        c = view(x, :, i)
        sum(logpdf_with_trans.(dist.v, c, istrans))
    end
end
function link(
    dist::VectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return mapvcat(dist.v, x) do dist, x
        link(dist, x)
    end
end
function link(
    dist::VectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x) do x
        link(dist, x)
    end
end

function invlink(
    dist::VectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return mapvcat(dist.v, x) do dist, x
        invlink(dist, x)
    end
end
function invlink(
    dist::VectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return eachcolmaphcat(x) do x
        invlink(dist, x)
    end
end

function logpdf_with_trans(
    dist::MatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return _sum(dist.dists, x) do dist, x
        logpdf_with_trans(dist, x, istrans)
    end
end
function logpdf_with_trans(
    dist::MatrixOfUnivariate,
    x::AbstractArray{<:AbstractMatrix{<:Real}},
    istrans::Bool,
)
    return mapvcat(x) do x
        logpdf_with_trans(dist, x, istrans)
    end
end
function link(
    dist::MatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return mapvcat(dist.dists, x) do dist, x
        link(dist, x)
    end
end
function invlink(
    dist::MatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return mapvcat(dist.dists, x) do dist, x
        invlink(dist, x)
    end
end
