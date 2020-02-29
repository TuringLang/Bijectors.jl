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
    _logpdf_with_trans_pd(d, X, transform)
end
function logpdf_with_trans(
    d::TuringWishart,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
    transform::Bool
)
    _logpdf_with_trans_pd.(Ref(d), X, transform)
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
    _logpdf_with_trans_pd.(Ref(d), X, transform)
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
    return sum(logpdf_with_trans.(dist.v.value, x, istrans))
end
function logpdf_with_trans(
    dist::FillVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    temp = vec(sum(reshape(logpdf_with_trans.(dist.v.value, x, istrans), size(x)), dims = 1))
    init = vcat(temp[1])
    return reduce(vcat, drop(temp, 1); init = init)
end
function link(
    dist::FillVectorOfUnivariate,
    x::AbstractVecOrMat{<:Real},
)
    return link(dist.v.value, x)
end
function invlink(
    dist::FillVectorOfUnivariate,
    x::AbstractVecOrMat{<:Real},
)
    return invlink(dist.v.value, x)
end

function logpdf_with_trans(
    dist::FillMatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans.(dist.dists.value, x, istrans))
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
    sum(logpdf_with_trans.(dist.dists, eachcol(x), istrans))
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
    @views init = reshape(link(dist.dists[1], x[:,1]), :, 1)
    return mapreduce((dist, c) -> link(dist, c), hcat, drop(dist.dists, 1), drop(eachcol(x), 1); init = init)
end
function invlink(
    dist::VectorOfMultivariate,
    x::AbstractMatrix{<:Real},
)
    @views init = reshape(invlink(dist.dists[1], x[:,1]), :, 1)
    return mapreduce((dist, c) -> invlink(dist, c), hcat, drop(dist.dists, 1), drop(eachcol(x), 1); init = init)
end

function logpdf_with_trans(
    dist::VectorOfUnivariate,
    x::AbstractVector{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans.(dist.dists, x, istrans))
end
function logpdf_with_trans(
    dist::VectorOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return mapvcat(eachcol(x)) do x
        logpdf_with_trans(dist, x, istrans)
    end
end
function link(
    dist::VectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return mapvcat(dist.dists, x) do dist, x
        link(dist, x)
    end
end
function link(
    dist::VectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return mapvcat(dist.dists, eachcol(x)) do dist, x
        link(dist, x)
    end
end

function logpdf_with_trans(
    dist::MatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    sum(logpdf_with_trans.(dist, x, istrans))
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
    return mapvcat(dist.dists, x) do x
        link(dist, x)
    end
end
function invlink(
    dist::MatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return mapvcat(dist.dists, x) do x
        invlink(dist, x)
    end
end
