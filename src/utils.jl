# `permutedims` seems to work better with AD (cf. KernelFunctions.jl)
aT_b(a::AbstractVector{<:Real}, b::AbstractMatrix{<:Real}) = permutedims(a) * b
# `permutedims` can't be used here since scalar output is desired
aT_b(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) = dot(a, b)

# flatten arrays with fallback for scalars
_vec(x::AbstractArray{<:Real}) = vec(x)
_vec(x::Real) = x

# Useful for reconstructing objects.
reconstruct(b, args...) = constructorof(typeof(b))(args...)

# Despite kwargs using `NamedTuple` in Julia 1.6, I'm still running
# into type-instability issues when using `eachslice`. So we define our own.
# https://github.com/JuliaLang/julia/issues/39639
# TODO: Check if this is the case in Julia 1.7.
# Adapted from https://github.com/JuliaLang/julia/blob/ef673537f8622fcaea92ac85e07962adcc17745b/base/abstractarraymath.jl#L506-L513.
# FIXME: It seems to break Zygote though. Which is weird because normal `eachslice` does not.
@inline function Base.eachslice(A::AbstractArray, ::Val{N}) where {N}
    dim = N
    dim <= ndims(A) || throw(DimensionMismatch("A doesn't have $dim dimensions"))
    inds_before = ntuple(_ -> Colon(), dim-1)
    inds_after = ntuple(_ -> Colon(), ndims(A)-dim)
    return (view(A, inds_before..., i, inds_after...) for i in axes(A, dim))
end
