using SparseArrays
using ArgCheck

"""
    Permute{A} <: Bijector{1}

A bijector implementation of a permutation. The permutation is performed
using a matrix of type `A`. There are a couple of different ways to construct `Permute`:

    Permute([0 1; 1 0])          # will map [1, 2] => [2, 1]
    Permute([2, 1])              # will map [1, 2] => [2, 1]
    Permute(2, 2 => 1, 1 => 2)   # will map [1, 2] => [2, 1]
    Permute(2, [1, 2] => [2, 1]) # will map [1, 2] => [2, 1]

If this is not clear, the examples might be of help.

# Examples
A simple example is permuting a vector of size 3.

```julia-repl
julia> b1 = Permute([
           0 1 0;
           1 0 0;
           0 0 1
       ])
Permute{Array{Int64,2}}([0 1 0; 1 0 0; 0 0 1])

julia> b2 = Permute([2, 1, 3])           # specify all elements at once
Permute{SparseArrays.SparseMatrixCSC{Float64,Int64}}(

  [2, 1]  =  1.0
  [1, 2]  =  1.0
  [3, 3]  =  1.0)

julia> b3 = Permute(3, 2 => 1, 1 => 2)    # element-wise
Permute{SparseArrays.SparseMatrixCSC{Float64,Int64}}(
  [2, 1]  =  1.0
  [1, 2]  =  1.0
  [3, 3]  =  1.0)

julia> b4 = Permute(3, [1, 2] => [2, 1])  # block-wise
Permute{SparseArrays.SparseMatrixCSC{Float64,Int64}}(
  [2, 1]  =  1.0
  [1, 2]  =  1.0
  [3, 3]  =  1.0)

julia> b1.A == b2.A == b3.A == b4.A
true

julia> b1([1., 2., 3.])
3-element Array{Float64,1}:
 2.0
 1.0
 3.0

julia> b2([1., 2., 3.])
3-element Array{Float64,1}:
 2.0
 1.0
 3.0

julia> b3([1., 2., 3.])
3-element Array{Float64,1}:
 2.0
 1.0
 3.0

julia> b4([1., 2., 3.])
3-element Array{Float64,1}:
 2.0
 1.0
 3.0

julia> inv(b1)
Permute{LinearAlgebra.Transpose{Int64,Array{Int64,2}}}([0 1 0; 1 0 0; 0 0 1])

julia> inv(b1)(b1([1., 2., 3.]))
3-element Array{Float64,1}:
 1.0
 2.0
 3.0
```
"""
struct Permute{A} <: Bijector{1}
    A::A
end

function Permute(indices::AbstractVector{Int})
    # construct a sparse-matrix for use in the multiplication
    n = length(indices)
    A = spzeros(n, n)

    for (i, idx) in enumerate(indices)
        A[idx, i] = 1.0
    end

    return Permute(A)
end

function Permute(n::Int, indices::Pair{Int, Int}...)
    A = spdiagm(0 => ones(n))

    dests = Set{Int}()
    sources = Set{Int}()

    for (src, dst) in indices
        @argcheck dst ∉ dests
        @argcheck src ∉ sources

        push!(dests, dst)
        push!(sources, src)
        
        A[dst, src] = 1.0
        A[src, src] = 0.0  # <= remove `src => src`
    end

    @argcheck (sources ∩ dests) == (sources ∪ dests) "$sources ∩ $dests ≠ $sources ∪ $dests"

    dropzeros!(A)
    return Permute(A)
end

function Permute(n::Int, indices::Pair{Vector{Int}, Vector{Int}}...)
    A = spdiagm(0 => ones(n))

    dests = Set{Int}()
    sources = Set{Int}()

    for (srcs, dsts) in indices
        @argcheck length(srcs) == length(dsts) "$srcs => $dsts is not bijective"
        
        for (src, dst) in zip(srcs, dsts)
            @argcheck dst ∉ dests "$dst used more than once"
            @argcheck src ∉ sources "$src used more than once"

            push!(dests, dst)
            push!(sources, src)
            
            A[dst, src] = 1.0
            A[src, src] = 0.0  # <= remove `src => src`
        end
    end

    @argcheck (sources ∩ dests) == (sources ∪ dests) "$sources ∩ $dests ≠ $sources ∪ $dests"

    dropzeros!(A)
    return Permute(A)
end


@inline (b::Permute)(x::AbstractVecOrMat) = b.A * x
@inline inv(b::Permute) = Permute(transpose(b.A))

logabsdetjac(b::Permute, x::AbstractVector) = zero(eltype(x))
logabsdetjac(b::Permute, x::AbstractMatrix) = zero(eltype(x), size(x, 2))
