using SparseArrays

"""
    PartitionMask{A}(A_1::A, A_2::A, A_3::A) where {A}

This is used to partition and recombine a vector into 3 disjoint "subvectors".

Implements
- `partition(m::PartitionMask, x)`: partitions `x` into 3 disjoint "subvectors"
- `combine(m::PartitionMask, x_1, x_2, x_3)`: combines 3 disjoint vectors into a single one

Note that `PartitionMask` is _not_ a `Bijector`. It is indeed a bijection, but
does not follow the `Bijector` interface.

Its main use is in `CouplingLayer` where we want to partition the input into 3 parts,
one part to transform, one part to map into the parameter-space of the transform applied
to the first part, and the last part of the vector is not used for anything.

# Examples
```julia-repl
julia> m = PartitionMask(3, [1], [2]) # <= assumes input-length 3
PartitionMask{SparseArrays.SparseMatrixCSC{Float64,Int64}}(
  [1, 1]  =  1.0,
  [2, 1]  =  1.0,
  [3, 1]  =  1.0)

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

"""
struct PartitionMask{A}
    A_1::A
    A_2::A
    A_3::A

    # Only make it possible to construct using matrices
    PartitionMask(A_1::A, A_2::A, A_3::A) where {A <: AbstractMatrix{<:Real}} = new{A}(A_1, A_2, A_3)
end

function PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int},
    indices_3::AbstractVector{Int}
)
    A_1 = spzeros(Bool, n, length(indices_1));
    A_2 = spzeros(Bool, n, length(indices_2));
    A_3 = spzeros(Bool, n, length(indices_3));

    for (i, idx) in enumerate(indices_1)
        A_1[idx, i] = true
    end

    for (i, idx) in enumerate(indices_2)
        A_2[idx, i] = true
    end

    for (i, idx) in enumerate(indices_3)
        A_3[idx, i] = true
    end

    return PartitionMask(A_1, A_2, A_3)
end

PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int}
) = PartitionMask(n, indices_1, indices_2, nothing)

PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int},
    indices_3::Nothing
) = PartitionMask(n, indices_1, indices_2, [i for i in 1:n if i ∉ (indices_1 ∪ indices_2)])

PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::Nothing,
    indices_3::AbstractVector{Int}
) = PartitionMask(n, indices_1, [i for i in 1:n if i ∉ (indices_1 ∪ indices_3)], indices_3)

"""
    PartitionMask(n::Int, indices)

Assumes you want to _split_ the vector, where `indices` refer to the 
parts of the vector you want to apply the bijector to.
"""
function PartitionMask(n::Int, indices)
    indices_2 = [i for i in 1:n if i ∉ indices]

    # sparse arrays <3
    A_1 = spzeros(Bool, n, length(indices));
    A_2 = spzeros(Bool, n, length(indices_2));

    # Like doing:
    #    A[1, 1] = 1.0
    #    A[3, 2] = 1.0
    for (i, idx) in enumerate(indices)
        A_1[idx, i] = true
    end

    for (i, idx) in enumerate(indices_2)
        A_2[idx, i] = true
    end

    return PartitionMask(A_1, A_2, spzeros(Bool, n, 0))
end
PartitionMask(x::AbstractVector, indices) = PartitionMask(length(x), indices)

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


# CouplingLayer

"""
    CouplingLayer{B, M, F}(mask::M, θ::F)

Implements a coupling-layer as defined in [1].

# Examples
```julia-repl
julia> m = PartitionMask(3, [1], [2]) # <= going to use x[2] to parameterize transform of x[1]
PartitionMask{SparseArrays.SparseMatrixCSC{Float64,Int64}}(
  [1, 1]  =  1.0, 
  [2, 1]  =  1.0, 
  [3, 1]  =  1.0)

julia> cl = CouplingLayer(Shift, m, identity) # <= will do `y[1:1] = x[1:1] + x[2:2]`;

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
struct CouplingLayer{B, M, F} <: Bijector{1} where {B, M <: PartitionMask, F}
    mask::M
    θ::F
end

CouplingLayer(B, mask::M) where {M} = CouplingLayer{B, M, typeof(identity)}(mask, identity)
CouplingLayer(B, mask::M, θ::F) where {M, F} = CouplingLayer{B, M, F}(mask, θ)
function CouplingLayer(B, θ, n::Int)
    idx = Int(floor(n / 2))
    return CouplingLayer(B, PartitionMask(n, 1:idx), θ)
end

function CouplingLayer(cl::CouplingLayer{B}, mask::PartitionMask) where {B}
    return CouplingLayer(B, mask, cl.θ)
end

"Returns the constructor of the coupling law."
coupling(cl::CouplingLayer{B}) where {B} = B

"Returns the coupling law constructed from `x`."
function couple(cl::CouplingLayer{B}, x::AbstractVector) where {B}
    # partition vector using `cl.mask::PartitionMask`
    x_1, x_2, x_3 = partition(cl.mask, x)

    # construct bijector `B` using θ(x₂)
    b = B(cl.θ(x_2))

    return b
end

function (cl::CouplingLayer{B})(x::AbstractVector) where {B}
    # partition vector using `cl.mask::PartitionMask`
    x_1, x_2, x_3 = partition(cl.mask, x)

    # construct bijector `B` using θ(x₂)
    b = B(cl.θ(x_2))

    # recombine the vector again using the `PartitionMask`
    return combine(cl.mask, b(x_1), x_2, x_3)
end
function (cl::CouplingLayer{B})(x::AbstractMatrix) where {B}
    return hcat([cl(x[:, i]) for i = 1:size(x, 2)]...)
end


function (icl::Inverse{<:CouplingLayer{B}})(y::AbstractVector) where {B}
    cl = icl.orig
    
    y_1, y_2, y_3 = partition(cl.mask, y)

    b = B(cl.θ(y_2))
    ib = inv(b)

    return combine(cl.mask, ib(y_1), y_2, y_3)
end
function (icl::Inverse{<:CouplingLayer{B}})(y::AbstractMatrix) where {B}
    return hcat([icl(y[:, i]) for i = 1:size(y, 2)]...)
end

function logabsdetjac(cl::CouplingLayer{B}, x::AbstractVector) where {B}
    x_1, x_2, x_3 = partition(cl.mask, x)
    b = B(cl.θ(x_2))

    # `B` might be 0-dim in which case it will treat `x_1` as a batch
    # therefore we sum to ensure such a thing does not happen
    return sum(logabsdetjac(b, x_1))
end

function logabsdetjac(cl::CouplingLayer{B}, x::AbstractMatrix) where {B}
    r = [logabsdetjac(cl, x[:, i]) for i = 1:size(x, 2)]

    # FIXME: this really needs to be handled in a better way
    # We need to return a `TrackedArray`
    if Tracker.istracked(r[1])
        return Tracker.collect(r)
    else
        return r
    end
end
