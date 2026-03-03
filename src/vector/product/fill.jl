# This file contains performance optimisation for homogeneous product distributions (e.g. d
# = product_distribution(fill(Beta(2, 2), N))). The idea is that when the vector bijector is
# determined solely by the type, we can compute the transform once and reuse it.
#
# For example when N = 10, the optimised path here brings from_linked_vec(d)(y) down 
# from 731 ns to 59 ns, and to_linked_vec(d)(x) from 278 ns to 75 ns.

"""
    Elementwise{T,M}

Similar in principle to a `FillArrays.Fill`: this represents an array of size `size` where
all entries are `value`. However, this does not subtype AbstractArray.

This is used to represent the vector bijector for homogeneous product distributions, where
the same transform is applied to each component. By using a `Elementwise`, we can compute
the transform once and reuse it, rather than computing it separately for each component.
"""
struct Elementwise{T,M}
    value::T
    size::Dims{M}

    Elementwise(value::T, size::NTuple{N,Int}) where {T,N} = new{T,N}(value, size)

    # Unwrap scalar-to-scalar bijectors
    function Elementwise(value::VectWrap{T}, size::NTuple{N,Int}) where {T,N}
        return new{T,N}(value.bijector, size)
    end
    function Elementwise(value::OnlyWrap{T}, size::NTuple{N,Int}) where {T,N}
        return new{T,N}(value.bijector, size)
    end
end
Base.:(==)(a::Elementwise, b::Elementwise) = (a.value == b.value) & (a.size == b.size)
function Base.isequal(a::Elementwise, b::Elementwise)
    return isequal(a.value, b.value) && isequal(a.size, b.size)
end

_map_inverse(t::Elementwise) = Elementwise(inverse(t.value), t.size)

# Trait: returns true when the vector bijector for a distribution type is determined
# solely by the type (not runtime parameter values).
_has_constant_vec_bijector(::Type) = false
_has_constant_vec_bijector(::Type{<:IDENTITY_UNIVARIATES}) = true
_has_constant_vec_bijector(::Type{<:POSITIVE_UNIVARIATES}) = true
# between 0 and 1
function _has_constant_vec_bijector(
    ::Type{<:Union{D.Beta,D.KSOneSided,D.NoncentralBeta,D.LogitNormal}}
)
    return true
end
_has_constant_vec_bijector(::Type{<:D.DiscreteUnivariateDistribution}) = true
# Multivariates
_has_constant_vec_bijector(::Type{<:D.AbstractMvNormal}) = true
_has_constant_vec_bijector(::Type{<:D.AbstractMvLogNormal}) = true
_has_constant_vec_bijector(::Type{<:SIMPLEX_MULTIVARIATES}) = true
_has_constant_vec_bijector(::Type{<:D.DiscreteMultivariateDistribution}) = true

function (t::ProductVecTransform{<:Elementwise,Nothing,Dims{N}})(
    x::AbstractArray{T,MplusN}
) where {N,T,MplusN}
    trf = t.transforms.value
    return if N == 0
        vec(trf.(x))
    else
        dims = ntuple(i -> i + N, Val(MplusN - N))
        vec(stack(map(trf, eachslice(x; dims=dims))))
    end
end

# Tiny struct that allows us to use map(eachslices) directly instead of a manual loop inside
# with_logabsdet_jacobian.
mutable struct WithLogabsdetjac{T,R}
    trf::T
    logjac::R
end
function (w::WithLogabsdetjac{T,R})(x::AbstractArray{N}) where {T,R,N}
    y, lj = with_logabsdet_jacobian(w.trf, x)
    # mutate the logjac in place; we'll pick it up later
    w.logjac += lj
    # output the thing that map expects
    return y
end

function with_logabsdet_jacobian(
    t::ProductVecTransform{<:Elementwise,Nothing,Dims{N}}, x::AbstractArray{T,MplusN}
) where {N,T,MplusN}
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
        dims = ntuple(i -> i + N, Val(MplusN - N))
        lj = WithLogabsdetjac(trf, _fzero(T))
        y = vec(stack(map(lj, eachslice(x; dims=dims))))
        y, lj.logjac
    end
end

function (t::ProductVecInvTransform{<:Elementwise{F,M},Nothing,Dims{N}})(
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
        stack(map(t.transforms.value, eachslice(reshaped_y; dims=dims)))
    end
end

function with_logabsdet_jacobian(
    t::ProductVecInvTransform{<:Elementwise{F,M},Nothing,Dims{N}}, y::AbstractVector{T}
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
        x = stack(map(lj, eachslice(reshaped_y; dims=dims)))
        x, lj.logjac
    end
end
