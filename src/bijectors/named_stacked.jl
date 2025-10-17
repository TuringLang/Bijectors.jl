"""
    NamedStacked(transforms::NamedTuple)
"""
struct NamedStacked{names,Ttrf<:NamedTuple{names},Trng<:NamedTuple{names}} <: Transform
    # This should be a NamedTuple of bijectors
    transforms::Ttrf
    # This should be a NamedTuple of UnitRanges OR integers.
    # UnitRanges are necessary when the output of a transform is not a scalar. If the output
    # is a scalar then an integer should be used.
    ranges::Trng

    function NamedStacked{names}(
        transforms::Ttrf, ranges::Trng
    ) where {names,Ttrf<:NamedTuple{names},Trng<:NamedTuple{names}}
        return new{names,Ttrf,Trng}(transforms, ranges)
    end
end

# Need to overload this or else it goes into a stack overflow between Inverse(b) and
# isinvertible(b)...
isinvertible(::NamedStacked) = true

@generated function bijector(
    d::Distributions.ProductNamedTupleDistribution{names}
) where {names}
    exprs = []
    push!(exprs, :(transforms = NamedTuple()))
    push!(exprs, :(ranges = NamedTuple()))
    push!(exprs, :(offset = 1))
    for n in names
        push!(exprs, :(dist = d.dists.$n))
        push!(exprs, :(trf = bijector(dist)))
        push!(exprs, :(output_sz_tuple = output_size(trf, size(dist))))
        push!(
            exprs,
            :(
                if length(output_sz_tuple) == 0
                    output_range = offset
                    offset += 1
                elseif length(output_sz_tuple) == 1
                    output_range = offset:(offset + only(output_sz_tuple) - 1)
                    offset += only(output_sz_tuple)
                else
                    errmsg = "output size for distribution $d must not be multidimensional"
                    throw(ArgumentError(errmsg))
                end
            ),
        )
        push!(exprs, :(transforms = merge(transforms, ($n=trf,))))
        push!(exprs, :(ranges = merge(ranges, ($n=output_range,))))
    end
    push!(exprs, :(return NamedStacked{names}(transforms, ranges)))
    return Expr(:block, exprs...)
end

@generated function transform(ns::NamedStacked{names}, x::NamedTuple{names}) where {names}
    exprs = []
    # TODO: Not a fan of initialising this as Float64 but otherwise not sure how to make it
    # type stable as it would depend on the type of the distributions?
    push!(exprs, :(output = Float64[]))
    for n in names
        push!(exprs, :(output = vcat(output, ns.transforms.$n(x.$n))))
    end
    push!(exprs, :(return output))
    return Expr(:block, exprs...)
end

@generated function transform(
    nsi::Inverse{<:NamedStacked{names}}, x::AbstractVector
) where {names}
    exprs = []
    push!(exprs, :(output = NamedTuple()))
    for n in names
        push!(
            exprs,
            :(
                output = merge(
                    output, ($n=inverse(nsi.orig.transforms.$n)(x[nsi.orig.ranges.$n]),)
                )
            ),
        )
    end
    push!(exprs, :(return output))
    return Expr(:block, exprs...)
end
