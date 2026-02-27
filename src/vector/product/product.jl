# Product distributions of tuples and arrays.

struct ProductVecTransform{TTrf,Trng,D}
    "A collection of vectorisation transforms, one for each component of the product
    distribution. These may either be `to_vec` or `to_linked_vec` transforms, which in turn
    determines the overall behaviour of this transform."
    transforms::TTrf
    "A collection of ranges which specify the output range for each component of the product
    distribution's bijectors."
    ranges::Trng
    "Size of each distribution inside the product distribution. Distributions.jl enforces
    that all distributions in a product distribution have the same size."
    base_size::D
end
struct ProductVecInvTransform{TTrf,Trng,D}
    "A collection of inverse vectorisation transforms, one for each component of the product
    distribution. These may either be `from_vec` or `from_linked_vec` transforms, which in
    turn determines the overall behaviour of this transform."
    transforms::TTrf
    "A collection of ranges which specify the input range for each component of the product
    distribution's bijectors."
    ranges::Trng
    "Size of each distribution inside the product distribution. Distributions.jl enforces
    that all distributions in a product distribution have the same size."
    base_size::D
end
function inverse(t::ProductVecTransform)
    return ProductVecInvTransform(inverse.(t.transforms), t.ranges, t.base_size)
end
function inverse(t::ProductVecInvTransform)
    return ProductVecTransform(inverse.(t.transforms), t.ranges, t.base_size)
end

# zero(T) but with fallback for non-numeric T.
_fzero(::Type{T}) where {T<:Number} = zero(T)
_fzero(@nospecialize(T)) = 0.0

"""
Return an object that can be iterated over to obtain the values for each distribution
in the product distribution.
"""
function _get_val_iterator(::ProductVecTransform{<:Any,<:Any,Tuple{}}, x::AbstractArray)
    # if the base_size is an empty tuple, then it's a univariate distribution,
    # and the values are just the elements of `x`.
    return x
end
function _get_val_iterator(
    ::ProductVecTransform{<:Any,<:Any,NTuple{N,Int}}, x::AbstractArray{T,MplusN}
) where {T,MplusN,N}
    # Multivariate case. The distribution itself has dimension N, and has been expanded
    # by M extra dimensions (e.g. fill(MvNormal(...), 3, 3, 3) would have M=3 and N=2).
    # In the sample, the N dimensions come first, followed by the M dimensions.
    M = MplusN - N
    dims = ntuple(i -> i + N, M)
    return eachslice(x; dims=dims)
end

# TODO(penelopeysm): The `xvec[r] .= xr` is inefficient. We could do better by having
# mutating versions of bijectors.
function with_logabsdet_jacobian(t::ProductVecTransform, x::AbstractArray{T}) where {T}
    total_length = sum(length, t.ranges)
    logjac = _fzero(T)
    y = Vector{T}(undef, total_length)
    val_iterator = _get_val_iterator(t, x)
    for (trf, r, val) in zip(t.transforms, t.ranges, val_iterator)
        xr, lj = with_logabsdet_jacobian(trf, val)
        y[r] .= xr
        logjac += lj
    end
    return y, logjac
end
function (t::ProductVecTransform)(x::AbstractArray{T}) where {T}
    total_length = sum(length, t.ranges)
    y = Vector{T}(undef, total_length)
    val_iterator = _get_val_iterator(t, x)
    for (trf, r, val) in zip(t.transforms, t.ranges, val_iterator)
        y[r] .= trf(val)
    end
    return y
end

@generated function _set_lastindex!(
    x::AbstractArray{T,N}, i::CartesianIndex{M}, val
) where {T,M,N}
    colons = fill(:, N - M)
    return quote
        x[$colons..., i.I...] = val
    end
end
# Generalisation of size to include tuples.
_sz(::NTuple{N,Any}) where {N} = (N,)
_sz(x::AbstractArray) = size(x)
# Generalisation of CartesianIndices to include tuples.
_cartesian_indices(::NTuple{N,Any}) where {N} = CartesianIndices((N,))
_cartesian_indices(x::AbstractArray) = CartesianIndices(x)
function with_logabsdet_jacobian(t::ProductVecInvTransform, y::AbstractVector{T}) where {T}
    x = Array{T}(undef, t.base_size..., _sz(t.transforms)...)
    logjac = _fzero(T)
    idxs = _cartesian_indices(t.transforms)
    for (idx, trf, r) in zip(idxs, t.transforms, t.ranges)
        xr, lj = with_logabsdet_jacobian(trf, view(y, r))
        _set_lastindex!(x, idx, xr)
        logjac += lj
    end
    return x, logjac
end
function (t::ProductVecInvTransform)(y::AbstractVector{T}) where {T}
    x = Array{T}(undef, t.base_size..., _sz(t.transforms)...)
    idxs = _cartesian_indices(t.transforms)
    for (idx, trf, r) in zip(idxs, t.transforms, t.ranges)
        xr = trf(view(y, r))
        _set_lastindex!(x, idx, xr)
    end
    return x
end

# These must be generated functions for type stability.
@generated function _make_transform(
    d::D.ProductDistribution{M,N,<:NTuple{NDists,D.Distribution}},
    indiv_transform_fn,
    length_fn,
    struct_type,
) where {M,N,NDists}
    exprs = []
    trfms = Expr(:tuple)
    for i in 1:NDists
        push!(trfms.args, :(indiv_transform_fn(d.dists[$i])))
    end
    push!(exprs, :(trfms = $trfms))
    push!(exprs, :(ranges = ()))
    push!(exprs, :(offset = 1))
    for i in 1:NDists
        push!(exprs, :(this_length = length_fn(d.dists[$i])))
        push!(exprs, :(ranges = (ranges..., offset:(offset + this_length - 1))))
        push!(exprs, :(offset += this_length))
    end
    push!(exprs, :(return struct_type(trfms, ranges, size(d.dists[1]))))
    return Expr(:block, exprs...)
end

function _make_transform(
    d::D.ProductDistribution{M,N,<:AbstractArray{<:D.Distribution}},
    indiv_transform_fn,
    length_fn,
    struct_type,
) where {M,N}
    trfms = map(indiv_transform_fn, d.dists)
    ranges = Array{UnitRange{Int}}(undef, size(d.dists)...)
    offset = 1
    for (i, dist) in enumerate(d.dists)
        this_length = length_fn(dist)
        ranges[i] = offset:(offset + this_length - 1)
        offset += this_length
    end
    return struct_type(trfms, ranges, size(d.dists[1]))
end

function from_vec(d::D.ProductDistribution)
    return _make_transform(d, from_vec, vec_length, ProductVecInvTransform)
end
function from_linked_vec(d::D.ProductDistribution)
    return _make_transform(d, from_linked_vec, linked_vec_length, ProductVecInvTransform)
end
function to_vec(d::D.ProductDistribution)
    return _make_transform(d, to_vec, vec_length, ProductVecTransform)
end
function to_linked_vec(d::D.ProductDistribution)
    return _make_transform(d, to_linked_vec, linked_vec_length, ProductVecTransform)
end

vec_length(d::D.ProductDistribution) = sum(vec_length, d.dists)
linked_vec_length(d::D.ProductDistribution) = sum(linked_vec_length, d.dists)

# This is conceptually what we need to do, but we need to also correct for the internal structure of the product distribution
# optic_vec(d::D.ProductDistribution) = mapreduce(optic_vec, vcat, d.dists)
# linked_optic_vec(d::D.ProductDistribution) = mapreduce(linked_optic_vec, vcat, d.dists)

append_index(::Nothing, i) = nothing
append_index(::AbstractPPL.Iden, i) = @opticof(_[i])
function append_index(p::AbstractPPL.Property{sym}, i) where {sym}
    return AbstractPPL.Property{sym}(append_index(p.child, i))
end
function append_index(p::AbstractPPL.Index, i)
    return if p.child isa AbstractPPL.Iden
        AbstractPPL.Index((p.ix..., i), p.kw, p.child)
    else
        AbstractPPL.Index(p.ix, p.kw, append_index(p.child, i))
    end
end

function optic_vec(d::D.ProductDistribution)
    optics = Union{}[]
    idxs = _cartesian_indices(d.dists)
    for (idx, dist) in zip(idxs, d.dists)
        this_dist_optics = optic_vec(dist)
        for optic in this_dist_optics
            optics = vcat(optics, append_index(optic, idx))
        end
    end
    return optics
end

function linked_optic_vec(d::D.ProductDistribution)
    optics = Union{}[]
    idxs = _cartesian_indices(d.dists)
    for (idx, dist) in zip(idxs, d.dists)
        this_dist_optics = linked_optic_vec(dist)
        for optic in this_dist_optics
            optics = vcat(optics, append_index(optic, idx))
        end
    end
    return optics
end
