"""
    NamedStacked{names}(transforms::NamedTuple, ranges::NamedTuple)

A bijector that contains a `NamedTuple` of bijectors. This is meant primarily for transforming
`Distributions.ProductNamedTupleDistribution` and samples from them.

The arguments `transforms` and `ranges` must be `NamedTuple`s with the same field names, and
these must also match the field names of the `ProductNamedTupleDistribution` that this bijector
corresponds to.

`ranges` specifies the index or indices in the output vector that correspond to the output of
each individual bijector in `transforms`. Its elements should be either `UnitRange`s or integers.
UnitRanges are necessary when the output of a transform is not a scalar. If the output is a
scalar then an integer should be used.

## Example

```jldoctest
julia> using Bijectors, LinearAlgebra

julia> d = Distributions.ProductNamedTupleDistribution((
           a = LogNormal(),
           b = InverseGamma(2, 3),
           c = MvNormal(zeros(2), I),
       ));

julia> b = bijector(d)
Bijectors.NamedStacked{(:a, :b, :c), @NamedTuple{a::Base.Fix1{typeof(broadcast), typeof(log)}, b::Base.Fix1{typeof(broadcast), typeof(log)}, c::typeof(identity)}, @NamedTuple{a::Int64, b::Int64, c::UnitRange{Int64}}}((a = Base.Fix1{typeof(broadcast), typeof(log)}(broadcast, log), b = Base.Fix1{typeof(broadcast), typeof(log)}(broadcast, log), c = identity), (a = 1, b = 2, c = 3:4))

julia> b.transforms.a == bijector(d.dists.a)
true

julia> x = (a = 1.0, b = 2.0, c = [0.5, -0.5]);

julia> y, logjac = with_logabsdet_jacobian(b, x)
([0.0, 0.6931471805599453, 0.5, -0.5], -0.6931471805599453)
```
"""
struct NamedStacked{names,Ttrf<:NamedTuple{names},Trng<:NamedTuple{names}} <: Transform
    # This should be a NamedTuple of bijectors
    transforms::Ttrf
    # This should be a NamedTuple of UnitRanges OR integers.
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

# Base.size doesn't work on ProductNamedTupleDistribution, so we need some custom machinery
# here. This enables us to nest PNTDists within each other.
# NOTE: For the outputs of this function to be correct, `trf` MUST be equal to
# bijector(dist).
function output_size(trf::NamedStacked, ::Distributions.ProductNamedTupleDistribution)
    return (sum(length, trf.ranges),)
end

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
        push!(exprs, :(output_sz_tuple = output_size(trf, dist)))
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
    # Note that `names` cannot be empty as `product_distribution(NamedTuple())` errors, so
    # we don't need to handle that case.
    for (i, n) in enumerate(names)
        if i == 1
            # need a vcat in case there's only one transform and it returns a scalar -- we
            # always want transform to return a vector.
            push!(exprs, :(output = vcat(ns.transforms.$n(x.$n))))
        else
            push!(exprs, :(output = vcat(output, ns.transforms.$n(x.$n))))
        end
    end
    push!(exprs, :(return output))
    return Expr(:block, exprs...)
end

@generated function with_logabsdet_jacobian(
    ns::NamedStacked{names}, x::NamedTuple{names}
) where {names}
    exprs = []
    # Note that `names` cannot be empty as `product_distribution(NamedTuple())` errors, so
    # we don't need to handle that case.
    for (i, n) in enumerate(names)
        if i == 1
            push!(
                exprs,
                quote
                    first_out, first_logjac = with_logabsdet_jacobian(
                        ns.transforms.$n, x.$n
                    )
                    output = vcat(first_out)
                    logjac = first_logjac
                end,
            )
        else
            push!(
                exprs,
                quote
                    next_out, next_logjac = with_logabsdet_jacobian(ns.transforms.$n, x.$n)
                    output = vcat(output, next_out)
                    logjac += next_logjac
                end,
            )
        end
    end
    push!(exprs, :(return output, logjac))
    return Expr(:block, exprs...)
end

@generated function transform(
    nsi::Inverse{<:NamedStacked{names}}, y::AbstractVector
) where {names}
    exprs = []
    push!(exprs, :(output = NamedTuple()))
    for (i, n) in enumerate(names)
        if i == 1
            push!(
                exprs,
                :(output = ($n=inverse(nsi.orig.transforms.$n)(y[nsi.orig.ranges.$n]),)),
            )
        else
            push!(
                exprs,
                :(
                    output = merge(
                        output, ($n=inverse(nsi.orig.transforms.$n)(y[nsi.orig.ranges.$n]),)
                    )
                ),
            )
        end
    end
    push!(exprs, :(return output))
    return Expr(:block, exprs...)
end

@generated function with_logabsdet_jacobian(
    nsi::Inverse{<:NamedStacked{names}}, y::AbstractVector
) where {names}
    exprs = []
    for (i, n) in enumerate(names)
        if i == 1
            push!(
                exprs,
                quote
                    first_out, first_logjac = with_logabsdet_jacobian(
                        inverse(nsi.orig.transforms.$n), y[nsi.orig.ranges.$n]
                    )
                    output = ($n=first_out,)
                    logjac = first_logjac
                end,
            )
        else
            push!(
                exprs,
                quote
                    next_out, next_logjac = with_logabsdet_jacobian(
                        inverse(nsi.orig.transforms.$n), y[nsi.orig.ranges.$n]
                    )
                    output = merge(output, ($n=next_out,))
                    logjac += next_logjac
                end,
            )
        end
    end
    push!(exprs, :(return output, logjac))
    return Expr(:block, exprs...)
end
