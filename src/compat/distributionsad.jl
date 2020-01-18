function logpdf_with_trans(
    d::TuringWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    _logpdf_with_trans_pd(d, X, transform)
end
function link(d::TuringWishart, X::AbstractMatrix{<:Real})
    _link_pd(d, X)
end
function invlink(d::TuringWishart, Y::AbstractMatrix{<:Real})
    _invlink_pd(d, Y)
end
function getlogp(d::TuringWishart, Xcf, X)
    return 0.5 * ((d.df - (dim(d) + 1)) * logdet(Xcf) - tr(d.chol \ X)) - d.c0
end

function logpdf_with_trans(
    d::TuringInverseWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    _logpdf_with_trans_pd(d, X, transform)
end
function link(d::TuringInverseWishart, X::AbstractMatrix{<:Real})
    _link_pd(d, X)
end
function invlink(d::TuringInverseWishart, Y::AbstractMatrix{<:Real})
    _invlink_pd(d, Y)
end
function getlogp(d::TuringInverseWishart, Xcf, X)
    Ψ = d.S
    return -0.5 * ((d.df + dim(d) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end
