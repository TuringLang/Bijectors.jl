using SparseArrays

"""
    PartitionMask{A}(A_1::A, A_2::A, A_3::A) where {A}

This is used to partition and recombine a vector into 3 disjoint "subvectors".

Implements
- `partition(m::PartitionMask, x)`: partitions `x` into 3 disjoint "subvectors"
- `combine(m::PartitionMask, x_1, x_2, x_3)`: combines 3 disjoint vectors into a single one

Note that `PartitionMask` is _not_ a `Bijector`. It is indeed a bijection, but
does not follow the `Bijector` interface.

Its main use is in `Coupling` where we want to partition the input into 3 parts,
one part to transform, one part to map into the parameter-space of the transform applied
to the first part, and the last part of the vector is not used for anything.

# Examples
```julia-repl
julia> using Bijectors: PartitionMask, partition, combine

julia> m = PartitionMask(3, [1], [2]) # <= assumes input-length 3
PartitionMask{Bool,SparseArrays.SparseMatrixCSC{Bool,Int64}}(
  [1, 1]  =  true, 
  [2, 1]  =  true, 
  [3, 1]  =  true)

julia> # Partition into 3 parts; the last part is inferred to be indices `[3, ]` from
       # the fact that `[1]` and `[2]` does not make up all indices in `1:3`.
       x1, x2, x3 = partition(m, [1., 2., 3.])
([1.0], [2.0], [3.0])

julia> # Recombines the partitions into a vector
       combine(m, x1, x2, x3)
3-element Array{Float64,1}:
 1.0
 2.0
 3.0
```
Note that the underlying `SparseMatrix` is using `Bool` as the element type. We can also
specify this to be some other type using the `sp_type` keyword:
```julia-repl
julia> m = PartitionMask{Float32}(3, [1], [2])
PartitionMask{Float32,SparseArrays.SparseMatrixCSC{Float32,Int64}}(
  [1, 1]  =  1.0, 
  [2, 1]  =  1.0, 
  [3, 1]  =  1.0)
```
"""
struct PartitionMask{T, A}
    A_1::A
    A_2::A
    A_3::A

    # Only make it possible to construct using matrices
    PartitionMask(A_1::A, A_2::A, A_3::A) where {T<:Real, A <: AbstractMatrix{T}} = new{T, A}(A_1, A_2, A_3)
end

PartitionMask(args...) = PartitionMask{Bool}(args...)

function PartitionMask{T}(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int},
    indices_3::AbstractVector{Int}
) where {T<:Real}
    A_1 = sparse(indices_1, 1:length(indices_1), one(T), n, length(indices_1))
    A_2 = sparse(indices_2, 1:length(indices_2), one(T), n, length(indices_2))
    A_3 = sparse(indices_3, 1:length(indices_3), one(T), n, length(indices_3))

    return PartitionMask(A_1, A_2, A_3)
end

PartitionMask{T}(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int};
) where {T} = PartitionMask{T}(n, indices_1, indices_2, nothing)

PartitionMask{T}(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int},
    indices_3::Nothing,
) where {T} = PartitionMask{T}(n, indices_1, indices_2, setdiff(1:n, indices_1, indices_2))

PartitionMask{T}(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::Nothing,
    indices_3::AbstractVector{Int},
) where {T} = PartitionMask{T}(n, indices_1, setdiff(1:n, indices_1, indices_3), indices_3)

"""
    PartitionMask(n::Int, indices)

Assumes you want to _split_ the vector, where `indices` refer to the 
parts of the vector you want to apply the bijector to.
"""
function PartitionMask{T}(n::Int, indices) where {T}
    indices_2 = setdiff(1:n, indices)

    # sparse arrays <3
    A_1 = sparse(indices, 1:length(indices), one(T), n, length(indices))
    A_2 = sparse(indices_2, 1:length(indices_2), one(T), n, length(indices_2))

    return PartitionMask(A_1, A_2, spzeros(T, n, 0))
end
function PartitionMask{T}(x::AbstractVector, indices) where {T}
    return PartitionMask{T}(length(x), indices)
end

"""
    combine(m::PartitionMask, x_1, x_2, x_3)

Combines `x_1`, `x_2`, and `x_3` into a single vector.
"""
@inline combine(m::PartitionMask, x_1, x_2, x_3) = m.A_1 * x_1 .+ m.A_2 * x_2 .+ m.A_3 * x_3

"""
    partition(m::PartitionMask, x)

Partitions `x` into 3 disjoint subvectors.
"""
@inline partition(m::PartitionMask, x) = (transpose(m.A_1) * x, transpose(m.A_2) * x, transpose(m.A_3) * x)


# Coupling

"""
    Coupling{F, M}(θ::F, mask::M)

Implements a coupling-layer as defined in [1].

# Examples
```julia-repl
julia> m = PartitionMask(3, [1], [2]) # <= going to use x[2] to parameterize transform of x[1]
PartitionMask{SparseArrays.SparseMatrixCSC{Float64,Int64}}(
  [1, 1]  =  1.0, 
  [2, 1]  =  1.0, 
  [3, 1]  =  1.0)

julia> cl = Coupling(θ -> Shift(θ[1]), m) # <= will do `y[1:1] = x[1:1] + x[2:2]`;

julia> x = [1., 2., 3.];

julia> cl(x)
3-element Array{Float64,1}:
 3.0
 2.0
 3.0

julia> inv(cl)(cl(x))
3-element Array{Float64,1}:
 1.0
 2.0
 3.0

julia> coupling(cl) # get the `Bijector` map `θ -> b(⋅, θ)`
Shift

julia> couple(cl, x) # get the `Bijector` resulting from `x`
Shift{Array{Float64,1},1}([2.0])
```

# References
[1] Kobyzev, I., Prince, S., & Brubaker, M. A., Normalizing flows: introduction and ideas, CoRR, (),  (2019). 
"""
struct Coupling{F, M} <: Bijector where {F, M <: PartitionMask}
    θ::F
    mask::M
end

function Coupling(θ, n::Int)
    idx = div(n, 2)
    return Coupling(θ, PartitionMask(n, 1:idx))
end

function Coupling(cl::Coupling, mask::PartitionMask)
    return Coupling(cl.θ, mask)
end

"Returns the constructor of the coupling law."
coupling(cl::Coupling) = cl.θ

"Returns the coupling law constructed from `x`."
function couple(cl::Coupling, x::AbstractVector)
    # partition vector using `cl.mask::PartitionMask`
    x_1, x_2, x_3 = partition(cl.mask, x)

    # construct bijector `B` using θ(x₂)
    b = cl.θ(x_2)

    return b
end

function transform(cl::Coupling, x::AbstractVector)
    # partition vector using `cl.mask::PartitionMask`
    x_1, x_2, x_3 = partition(cl.mask, x)

    # construct bijector `B` using θ(x₂)
    b = cl.θ(x_2)

    # recombine the vector again using the `PartitionMask`
    return combine(cl.mask, b(x_1), x_2, x_3)
end

function transform(icl::Inverse{<:Coupling}, y::AbstractVector)
    cl = icl.orig
    
    y_1, y_2, y_3 = partition(cl.mask, y)

    b = cl.θ(y_2)
    ib = inv(b)

    return combine(cl.mask, ib(y_1), y_2, y_3)
end

function logabsdetjac(cl::Coupling, x::AbstractVector)
    x_1, x_2, x_3 = partition(cl.mask, x)
    b = cl.θ(x_2)

    # `B` might be 0-dim in which case it will treat `x_1` as a batch
    # therefore we sum to ensure such a thing does not happen
    return sum(logabsdetjac(b, x_1))
end
