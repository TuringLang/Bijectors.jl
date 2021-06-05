# `permutedims` seems to work better with AD (cf. KernelFunctions.jl)
aT_b(a::AbstractVector{<:Real}, b::AbstractMatrix{<:Real}) = permutedims(a) * b
# `permutedims` can't be used here since scalar output is desired
aT_b(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) = dot(a, b)

# flatten arrays with fallback for scalars
_vec(x::AbstractArray{<:Real}) = vec(x)
_vec(x::Real) = x

# Useful for reconstructing objects.
reconstruct(b, args...) = constructorof(typeof(b))(args...)
