"""
    Reshape(in_shape, out_shape)

A [`Bijector`](@ref) that reshapes the input to the output shape.

# Example

```jldoctest
julia> using Bijectors: Reshape

julia> b = Reshape((2, 3), (3, 2))
Reshape{Tuple{Int64, Int64}, Tuple{Int64, Int64}}((2, 3), (3, 2))

julia> Array(transform(b, reshape(1:6, 2, 3)))
3×2 Matrix{Int64}:
 1  4
 2  5
 3  6
"""
struct Reshape{S1,S2} <: Bijector
    in_shape::S1
    out_shape::S2
end

inverse(b::Reshape) = Reshape(b.out_shape, b.in_shape)

with_logabsdet_jacobian(::Reshape, x) = reshape(x, b.out_shape), zero(eltype(x))
