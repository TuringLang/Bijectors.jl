struct ProductBijector{Bs,N} <: Transform
    bs::Bs
end

ProductBijector(bs::AbstractArray{<:Any,N}) where {N} = ProductBijector{typeof(bs),N}(bs)

inverse(b::ProductBijector) = ProductBijector(map(inverse, b.bs))

function _product_bijector_check_dim(::Val{N}, ::Val{M}) where {N,M}
    if N > M
        throw(
            DimensionMismatch(
                "Number of bijectors needs to be smaller than or equal to the number of dimensions",
            ),
        )
    end
end

function _product_bijector_slices(
    ::ProductBijector{<:AbstractArray,N}, x::AbstractArray{<:Real,M}
) where {N,M}
    _product_bijector_check_dim(Val(N), Val(M))

    # If N < M, then the bijectors expect an input vector of dimension `M - N`.
    # To achieve this, we need to slice along the last `N` dimensions.
    return eachslice(x; dims=ntuple(i -> i + (M - N), N))
end

# Specialization for case where we're just applying elementwise.
function transform(
    b::ProductBijector{<:AbstractArray,N}, x::AbstractArray{<:Real,N}
) where {N}
    return map(transform, b.bs, x)
end
# General case.
function transform(
    b::ProductBijector{<:AbstractArray,N}, x::AbstractArray{<:Real,M}
) where {N,M}
    slices = _product_bijector_slices(b, x)
    return stack(map(transform, b.bs, slices))
end

function with_logabsdet_jacobian(
    b::ProductBijector{<:AbstractArray,N}, x::AbstractArray{<:Real,N}
) where {N}
    results = map(with_logabsdet_jacobian, b.bs, x)
    return map(first, results), sum(last, results)
end
function with_logabsdet_jacobian(
    b::ProductBijector{<:AbstractArray,N}, x::AbstractArray{<:Real,M}
) where {N,M}
    slices = _product_bijector_slices(b, x)
    results = map(with_logabsdet_jacobian, b.bs, slices)
    return stack(map(first, results)), sum(last, results)
end

# Other utilities.
function output_size(b::ProductBijector{<:AbstractArray,N}, sz::NTuple{M}) where {N,M}
    _product_bijector_check_dim(Val(N), Val(M))

    sz_redundant = ntuple(i -> sz[i + (M - N)], N)
    sz_example = ntuple(i -> sz[i], M - N)
    # NOTE: `Base.stack`, which is used in the transformation, only supports the scenario where
    # all `b.bs` have the same output sizes => only need to check the first one.
    return (output_size(first(b.bs), sz_example)..., sz_redundant...)
end
