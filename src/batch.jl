abstract type AbstractBatch{T} <: AbstractVector{T} end

"""
    value(x)

Returns the underlying storage used for the entire batch.

If `x` is not `AbstractBatch`, then this is the identity function.
"""
value(x) = x

struct Batch{V, T} <: AbstractBatch{T}
    value::V
end

value(x::Batch) = x.value

# Convenient aliases
const ArrayBatch{N} = Batch{<:AbstractArray{<:Real, N}}
const VectorBatch = Batch{<:AbstractVector{<:AbstractArray{<:Real}}}

# Constructor for `ArrayBatch`.
Batch(x::AbstractVector{<:Real}) = Batch{typeof(x), eltype(x)}(x)
function Batch(x::AbstractArray{<:Real})
    V = typeof(x)
    # HACK: This assumes the batch is non-empty.
    T = typeof(getindex_for_last(x, 1))
    return Batch{V, T}(x)
end

# Constructor for `VectorBatch`.
Batch(x::AbstractVector{<:AbstractArray}) = Batch{typeof(x), eltype(x)}(x)

# `AbstractVector` interface.
Base.size(batch::Batch{<:AbstractArray{<:Any, N}}) where {N} = (size(value(batch), N), )

# Since impl inherited from `AbstractVector` doesn't do exactly what we want.
Base.similar(b::AbstractBatch) = reconstruct(b, similar(value(b)))

# For `VectorBatch`
Base.getindex(batch::VectorBatch, i::Int) = value(batch)[i]
Base.getindex(batch::VectorBatch, i::CartesianIndex{1}) = value(batch)[i]
Base.getindex(batch::VectorBatch, i) = Batch(value(batch)[i])
function Base.setindex!(batch::VectorBatch, v, i)
    # `v` can also be a `Batch`.
    Base.setindex!(value(batch), value(v), i)
    return batch
end

# For `ArrayBatch`
@generated function getindex_for_last(x::AbstractArray{<:Any, N}, inds) where {N}
    e = Expr(:call)
    push!(e.args, :(Base.view))
    push!(e.args, :x)

    for i = 1:N - 1
        push!(e.args, :(:))
    end

    push!(e.args, :(inds))

    return e
end

@generated function setindex_for_last!(out::AbstractArray{<:Any, N}, x, inds) where {N}
    e = Expr(:call)
    push!(e.args, :(Base.setindex!))
    push!(e.args, :out)
    push!(e.args, :x)

    for i = 1:N - 1
        push!(e.args, :(:))
    end

    push!(e.args, :(inds))

    return e
end

# General arrays.
Base.getindex(batch::ArrayBatch, i::Int) = getindex_for_last(value(batch), i)
Base.getindex(batch::ArrayBatch, i::CartesianIndex{1}) = getindex_for_last(value(batch), i)
Base.getindex(batch::ArrayBatch, i) = Batch(getindex_for_last(value(batch), i))
function Base.setindex!(batch::ArrayBatch, v, i)
    # `v` can also be a `Batch`.
    setindex_for_last!(value(batch), value(v), i)
    return batch
end
