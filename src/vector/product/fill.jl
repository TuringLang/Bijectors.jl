# This file contains performance optimisation for homogeneous product distributions (e.g. d
# = product_distribution(fill(Beta(2, 2), N))). The idea is that when the vector bijector is
# determined solely by the type, we can compute the transform once and reuse it.
#
# For example when N = 10, the optimised path here brings from_linked_vec(d)(y) down 
# from 731 ns to 59 ns, and to_linked_vec(d)(x) from 278 ns to 75 ns.

using FillArrays: FillArrays

"""
    Elementwise{T,S}

Similar in principle to a `FillArrays.Fill`: this represents an array of size `size` where
all entries are `value`. However, this does not subtype AbstractArray.

This is used to represent the vector bijector for homogeneous product distributions, where
the same transform is applied to each component. By using a `Elementwise`, we can compute
the transform once and reuse it, rather than computing it separately for each component.
"""
struct Elementwise{T,S}
    value::T
    size::S

    function Elementwise(value::T, size::S) where {N,T,S<:NTuple{N,Int}}
        return new{T,S}(value, size)
    end

    # Unwrap scalar-to-scalar bijectors
    function Elementwise(value::VectWrap{T}, size::S) where {N,T,S<:NTuple{N,Int}}
        return new{T,S}(value.bijector, size)
    end
    function Elementwise(value::OnlyWrap{T}, size::S) where {N,T,S<:NTuple{N,Int}}
        return new{T,S}(value.bijector, size)
    end
end
Base.:(==)(a::Elementwise, b::Elementwise) = (a.value == b.value) & (a.size == b.size)
function Base.isequal(a::Elementwise, b::Elementwise)
    return isequal(a.value, b.value) && isequal(a.size, b.size)
end

_map_inverse(t::Elementwise) = Elementwise(inverse(t.value), t.size)

"""
    Bijectors.VectorBijectors.has_constant_vec_bijector(::Type{T}) where {T}

Return `true` if the vector bijector for each element of a collection of distributions is
determined solely by the type of the distribution, and not by any runtime parameter
values.

This is slightly confusing, so is best explained by example. Consider

```julia
d = product_distribution(array_of_dists)
```

If it can be inferred from `typeof(array_of_dists)` that each distribution inside
`array_of_dists` has the same vector bijector, then
`has_constant_vec_bijector(typeof(array_of_dists))` should return `true`.

For example, if `array_of_dists` is a `FillArrays.Fill` of some distribution type, then we
know that each distribution inside is the same, and so they all have the same vector
bijector. Thus, we have that

```julia
has_constant_vec_bijector(::Type{<:FillArrays.Fill}) == true
```

For generic `AbstractArray`s or `Tuple`s, this will dispatch on the element type of the
array. That means that if a `dist::D` (where `D<:Distribution`) has a constant vector
bijector, we can simply mark `has_constant_vec_bijector(::Type{D}) == true`.

For example, `Beta` has a constant vector bijector, because its support is always between 0
and 1, regardless of its parameters.

On the other hand, `Uniform` does not have a constant vector bijector, because its support
depends on its parameters.
"""
has_constant_vec_bijector(::Type{TFill}) where {TFill<:FillArrays.Fill} = true
function has_constant_vec_bijector(::Type{<:AbstractArray{T}}) where {T}
    return has_constant_vec_bijector(T)
end
has_constant_vec_bijector(t::Type{<:Tuple}) = has_constant_vec_bijector(eltype(t))
has_constant_vec_bijector(::Type) = false
has_constant_vec_bijector(::Type{<:IDENTITY_UNIVARIATES}) = true
has_constant_vec_bijector(::Type{<:POSITIVE_UNIVARIATES}) = true
# between 0 and 1
function has_constant_vec_bijector(
    ::Type{<:Union{D.Beta,D.KSOneSided,D.NoncentralBeta,D.LogitNormal}}
)
    return true
end
has_constant_vec_bijector(::Type{<:D.DiscreteUnivariateDistribution}) = true
# Multivariates
has_constant_vec_bijector(::Type{<:D.AbstractMvNormal}) = true
has_constant_vec_bijector(::Type{<:D.AbstractMvLogNormal}) = true
has_constant_vec_bijector(::Type{<:SIMPLEX_MULTIVARIATES}) = true
has_constant_vec_bijector(::Type{<:D.DiscreteMultivariateDistribution}) = true

function (t::ProductVecTransform{<:Elementwise{F,Dims{M}},Nothing,Dims{N}})(
    x::AbstractArray{T}
) where {F,M,N,T}
    trf = t.transforms.value
    return if N == 0
        vec(trf.(x))
    else
        dims = ntuple(i -> i + N, Val(M))
        vec(stack(trf, eachslice(x; dims=dims)))
    end
end
function (::ProductVecTransform{<:Elementwise{TypedIdentity},Nothing,Dims{0}})(
    x::AbstractArray
)
    # If the wrapped transform is TypedIdentity, and the distribution is univariate (i.e.,
    # the 'base size' is `()::Dims{0}`), then the entire vectorisation transform amounts to
    # just `vec`. This special case is hit for things like
    # product_distribution(fill(Cauchy(), m1, m2, ...)).
    return vec(x)
end

# Tiny struct that allows us to use map(eachslices) directly instead of a manual loop inside
# with_logabsdet_jacobian.
mutable struct WithLogabsdetjac{T,R}
    trf::T
    logjac::R
end
function (w::WithLogabsdetjac)(x::AbstractArray)
    y, lj = with_logabsdet_jacobian(w.trf, x)
    # mutate the logjac in place; we'll pick it up later
    w.logjac += lj
    # output the thing that map expects
    return y
end

function with_logabsdet_jacobian(
    t::ProductVecTransform{<:Elementwise{F,Dims{M}},Nothing,Dims{N}}, x::AbstractArray{T}
) where {F,M,N,T}
    trf = t.transforms.value
    return if N == 0
        # univariate. This is a tiny bit faster than the mutable WithLogabsdetjac approach
        y = Vector{T}(undef, length(x))
        logjac = _fzero(T)
        for i in eachindex(x)
            y[i], lj = with_logabsdet_jacobian(trf, x[i])
            logjac += lj
        end
        y, logjac
    else
        dims = ntuple(i -> i + N, Val(M))
        lj = WithLogabsdetjac(trf, _fzero(T))
        y = vec(stack(lj, eachslice(x; dims=dims)))
        y, lj.logjac
    end
end
function with_logabsdet_jacobian(
    ::ProductVecTransform{<:Elementwise{TypedIdentity},Nothing,Dims{0}}, x::AbstractArray{T}
) where {T}
    return vec(x), _fzero(T)
end

function (t::ProductVecInvTransform{<:Elementwise{F,Dims{M}},Nothing,Dims{N}})(
    y::AbstractVector{T}
) where {F,M,N,T}
    return if N == 0
        # univariate -- we just need to apply the transform to everything, and
        # then reshape back into the original shape
        reshape(t.transforms.value.(y), t.transforms.size)
    else
        # dim 1 is the input for the inverse transform
        reshaped_y = reshape(y, :, t.transforms.size...)
        dims = ntuple(i -> i + 1, Val(M))
        stack(t.transforms.value, eachslice(reshaped_y; dims=dims))
    end
end
function (t::ProductVecInvTransform{<:Elementwise{TypedIdentity},Nothing,Dims{0}})(
    y::AbstractVector
)
    return reshape(y, t.transforms.size)
end

function with_logabsdet_jacobian(
    t::ProductVecInvTransform{<:Elementwise{F,Dims{M}},Nothing,Dims{N}},
    y::AbstractVector{T},
) where {F,M,N,T}
    trf = t.transforms.value
    return if N == 0
        # univariate
        x = Array{T}(undef, t.transforms.size)
        logjac = _fzero(T)
        for i in eachindex(y)
            x[i], lj = with_logabsdet_jacobian(trf, y[i])
            logjac += lj
        end
        x, logjac
    else
        reshaped_y = reshape(y, :, t.transforms.size...)
        dims = ntuple(i -> i + 1, Val(M))
        lj = WithLogabsdetjac(trf, _fzero(T))
        x = stack(lj, eachslice(reshaped_y; dims=dims))
        x, lj.logjac
    end
end
function with_logabsdet_jacobian(
    t::ProductVecInvTransform{<:Elementwise{TypedIdentity},Nothing,Dims{0}},
    y::AbstractVector{T},
) where {T}
    return reshape(y, t.transforms.size), _fzero(T)
end
