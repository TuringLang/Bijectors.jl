# Product distributions of tuples and arrays.

struct ProductDistributionVec{TTrf,Trng,D}
    "A collection of transforms, one for each component of the product distribution."
    transforms::TTrf
    "A collection of ranges which specify the output range for each component of the product
    distribution's bijectors."
    ranges::Trng
    "Size of each distribution inside the product distribution. Distributions.jl enforces
    that all distributions in a product distribution have the same size."
    base_size::D
end

function (t::ProductDistributionVec{<:Tuple,<:Tuple})(
    x::AbstractArray{T,MplusN}
) where {T,MplusN}
    total_length = sum(length.(t.ranges))
    xvec = Vector{T}(undef, total_length)
    if t.base_size == ()
        # Univariate base distribution
        for (trf, r, val) in zip(t.transforms, t.ranges, x)
            # TODO(penelopeysm): This is potentially inefficient since `trf(val)` always
            # allocates a new vector -- we probably want to at some point expand the API
            # to allow transformations to write in-place.
            xvec[r] .= trf(val)
        end
    else
        # Multivariate base distribution
        N = length(t.base_size)
        M = MplusN - N
        dims = ntuple(i -> i + M, N)
        for (trf, r, val) in zip(t.transforms, t.ranges, eachslice(x; dims=dims))
            xvec[r] .= trf(val)
        end
    end
    return xvec
end

# Must be generated functions for type stability
@generated function to_vec(
    d::D.ProductDistribution{M,N,<:NTuple{NDists,D.Distribution}}
) where {M,N,NDists}
    exprs = []
    trfms = Expr(:tuple)
    for i in 1:NDists
        push!(trfms.args, :(to_vec(d.dists[$i])))
    end
    push!(exprs, :(trfms = $trfms))
    push!(exprs, :(ranges = ()))
    push!(exprs, :(offset = 1))
    for i in 1:NDists
        push!(exprs, :(this_length = vec_length(d.dists[$i])))
        push!(exprs, :(ranges = (ranges..., offset:(offset + this_length - 1))))
        push!(exprs, :(offset += this_length))
    end
    push!(exprs, :(return ProductDistributionVec(trfms, ranges, size(d.dists[1]))))
    return Expr(:block, exprs...)
end

@generated function to_linked_vec(
    d::D.ProductDistribution{M,N,<:NTuple{NDists,D.Distribution}}
) where {M,N,NDists}
    exprs = []
    trfms = Expr(:tuple)
    for i in 1:NDists
        push!(trfms.args, :(to_linked_vec(d.dists[$i])))
    end
    push!(exprs, :(trfms = $trfms))
    push!(exprs, :(ranges = ()))
    push!(exprs, :(offset = 1))
    for i in 1:NDists
        push!(exprs, :(this_length = linked_vec_length(d.dists[$i])))
        push!(exprs, :(ranges = (ranges..., offset:(offset + this_length - 1))))
        push!(exprs, :(offset += this_length))
    end
    push!(exprs, :(return ProductDistributionVec(trfms, ranges, size(d.dists[1]))))
    return Expr(:block, exprs...)
end

# from_vec(d::D.MatrixDistribution) = Reshape(size(d))
# vec_length(d::D.MatrixDistribution) = prod(size(d))
# function optic_vec(d::D.MatrixDistribution)
#     return map(c -> AbstractPPL.Index(c.I, (;)), vec(CartesianIndices(size(d))))
# end
