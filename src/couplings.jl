using SparseArrays

# TODO: should we add another field `A_3` which we can use to filter out those
# parts of the vector to which we apply the identity? E.g.
# you want to use x[1] to parameterize transform of x[2], but you don't want
# to do anything with x[3]
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

@inline combine(m::PartitionMask, x_1, x_2, x_3) = m.A_1 * x_1 .+ m.A_2 * x_2 .+ m.A_3 * x_3
@inline partition(m::PartitionMask, x) = (transpose(m.A_1) * x, transpose(m.A_2) * x, transpose(m.A_3) * x)


# CouplingLayer

struct CouplingLayer{B, M, F} <: Bijector where {B, M <: PartitionMask, F}
    mask::M
    θ::F
end

CouplingLayer(B, mask::M, θ::F) where {M, F} = CouplingLayer{B, M, F}(mask, θ)
function CouplingLayer(B, θ, n::Int)
    idx = Int(floor(n / 2))
    return CouplingLayer(B, PartitionMask(n, 1:idx), θ)
end



function (cl::CouplingLayer{B})(x) where {B}
    x_1, x_2, x_3 = partition(cl.mask, x)

    b = B(cl.θ(x_2))

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
