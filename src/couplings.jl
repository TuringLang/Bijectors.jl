using SparseArrays

"""
    PartitionMask{A}(A_1::A, A_2::A, A_3::A) where {A}

This is used to partition and recombine a vector into 3 disjoint "subvectors".

Implements
- `partition(m::PartitionMask, x)`: partitions `x` into 3 disjoint "subvectors"
- `combine(m::PartitionMask, x_1, x_2, x_3)`: combines 3 disjoint vectors into a single one
"""
struct PartitionMask{A}
    A_1::A
    A_2::A
    A_3::A
end

function PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int},
    indices_3::AbstractVector{Int}
)
    A_1 = spzeros(n, length(indices_1));
    A_2 = spzeros(n, length(indices_2));
    A_3 = spzeros(n, length(indices_3));

    for (i, idx) in enumerate(indices_1)
        A_1[idx, i] = 1.0
    end

    for (i, idx) in enumerate(indices_2)
        A_2[idx, i] = 1.0
    end

    for (i, idx) in enumerate(indices_3)
        A_3[idx, i] = 1.0
    end

    return PartitionMask(A_1, A_2, A_3)
end

PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int},
    indices_3::Nothing
) = PartitionMask(n, indices_1, indices_2, Int[])

PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::Nothing,
    indices_3::AbstractVector{Int}
) = PartitionMask(n, indices_1, Int[], indices_3)

"""
    PartitionMask(n::Int, indices)

Assumes you want to _split_ the vector, where `indices` refer to the 
parts of the vector you want to apply the bijector to.
"""
function PartitionMask(n::Int, indices)
    indices_2 = [i for i in 1:n if i ∉ indices]

    # sparse arrays <3
    A_1 = spzeros(n, length(indices));
    A_2 = spzeros(n, length(indices_2));

    # Like doing:
    #    A[1, 1] = 1.0
    #    A[3, 2] = 1.0
    for (i, idx) in enumerate(indices)
        A_1[idx, i] = 1.0
    end

    for (i, idx) in enumerate(indices_2)
        A_2[idx, i] = 1.0
    end

    return PartitionMask(A_1, A_2, spzeros(n, 0))
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

# References
[1] Kobyzev, I., Prince, S., & Brubaker, M. A., Normalizing flows: introduction and ideas, CoRR, (),  (2019). 
"""
struct CouplingLayer{B, M, F} <: Bijector where {B, M <: PartitionMask, F}
    mask::M
    θ::F
end

CouplingLayer(B, mask::M, θ::F) where {M, F} = CouplingLayer{B, M, F}(mask, θ)
function CouplingLayer(B, θ, n::Int)
    idx = Int(floor(n / 2))
    return CouplingLayer(B, PartitionMask(n, 1:idx), θ)
end

function CouplingLayer(cl::CouplingLayer{B}, mask::PartitionMask) where {B}
    return CouplingLayer(B, mask, cl.θ)
end

function (cl::CouplingLayer{B})(x) where {B}
    # partition vector using `cl.mask::PartitionMask`
    x_1, x_2, x_3 = partition(cl.mask, x)

    # construct bijector `B` using θ(x₂)
    b = B(cl.θ(x_2))

    # recombine the vector again using the `PartitionMask`
    return combine(cl.mask, b(x_1), x_2, x_3)
end


function (icl::Inversed{<:CouplingLayer{B}})(y) where {B}
    cl = icl.orig
    
    y_1, y_2, y_3 = partition(cl.mask, y)

    b = B(cl.θ(y_2))
    ib = inv(b)

    return combine(cl.mask, ib(y_1), y_2, y_3)
end


function logabsdetjac(cl::CouplingLayer{B}, x) where {B}
    x_1, x_2, x_3 = partition(cl.mask, x)
    b = B(cl.θ(x_2))

    return logabsdetjac(b, x_1)
end
