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
function _get_val_iterator(::ProductVecTransform{<:Tuple,<:Tuple,Tuple{}}, x::AbstractArray)
    # if the base_size is an empty tuple, then it's a univariate distribution,
    # and the values are just the elements of `x`.
    return x
end
function _get_val_iterator(
    ::ProductVecTransform{<:Tuple,<:Tuple,NTuple{N,Int}}, x::AbstractArray{T,MplusN}
) where {T,MplusN,N}
    # Multivariate case.
    M = MplusN - N
    dims = ntuple(i -> i + M, N)
    return eachslice(x; dims=dims)
end

# TODO(penelopeysm): The `xvec[r] .= xr` is inefficient. We could do better by having
# mutating versions of bijectors.
function with_logabsdet_jacobian(
    t::ProductVecTransform{<:Tuple,<:Tuple}, x::AbstractArray{T}
) where {T}
    total_length = sum(length.(t.ranges))
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
function (t::ProductVecTransform{<:Tuple,<:Tuple})(x::AbstractArray{T}) where {T}
    total_length = sum(length.(t.ranges))
    y = Vector{T}(undef, total_length)
    val_iterator = _get_val_iterator(t, x)
    for (trf, r, val) in zip(t.transforms, t.ranges, val_iterator)
        y[r] .= trf(val)
    end
    return y
end

@generated function _set_lastindex!(x::AbstractArray{T,N}, i::Int, val) where {T,N}
    colons = fill(:, N - 1)
    return quote
        x[$colons..., i] = val
    end
end
function with_logabsdet_jacobian(
    t::ProductVecInvTransform{<:Tuple,<:Tuple}, y::AbstractVector{T}
) where {T}
    ndists = length(t.transforms)
    x = Array{T}(undef, t.base_size..., ndists)
    logjac = _fzero(T)
    for (i, (trf, r)) in enumerate(zip(t.transforms, t.ranges))
        xr, lj = with_logabsdet_jacobian(trf, view(y, r))
        _set_lastindex!(x, i, xr)
        logjac += lj
    end
    return x, logjac
end
function (t::ProductVecInvTransform{<:Tuple,<:Tuple})(y::AbstractVector{T}) where {T}
    ndists = length(t.transforms)
    x = Array{T}(undef, t.base_size..., ndists)
    for (i, (trf, r)) in enumerate(zip(t.transforms, t.ranges))
        xr = trf(view(y, r))
        _set_lastindex!(x, i, xr)
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
function from_vec(
    d::D.ProductDistribution{M,N,<:NTuple{NDists,D.Distribution}}
) where {M,N,NDists}
    return _make_transform(d, from_vec, vec_length, ProductVecInvTransform)
end
function from_linked_vec(
    d::D.ProductDistribution{M,N,<:NTuple{NDists,D.Distribution}}
) where {M,N,NDists}
    return _make_transform(d, from_linked_vec, linked_vec_length, ProductVecInvTransform)
end
function to_vec(
    d::D.ProductDistribution{M,N,<:NTuple{NDists,D.Distribution}}
) where {M,N,NDists}
    return _make_transform(d, to_vec, vec_length, ProductVecTransform)
end
function to_linked_vec(
    d::D.ProductDistribution{M,N,<:NTuple{NDists,D.Distribution}}
) where {M,N,NDists}
    return _make_transform(d, to_linked_vec, linked_vec_length, ProductVecTransform)
end

vec_length(d::D.ProductDistribution) = sum(vec_length, d.dists)
linked_vec_length(d::D.ProductDistribution) = sum(linked_vec_length, d.dists)

# This is conceptually what we need to do, but we need to also correct for the internal structure of the product distribution
# optic_vec(d::D.ProductDistribution) = mapreduce(optic_vec, vcat, d.dists)
# linked_optic_vec(d::D.ProductDistribution) = mapreduce(linked_optic_vec, vcat, d.dists)

optic_vec(d::D.ProductDistribution) = fill(nothing, vec_length(d))
linked_optic_vec(d::D.ProductDistribution) = fill(nothing, linked_vec_length(d))
