"""
    Stacked(bs)
    Stacked(bs, ranges)
    stack(bs::Bijector...)

A `Bijector` which stacks bijectors together which can then be applied to a vector
where `bs[i]::Bijector` is applied to `x[ranges[i]]::UnitRange{Int}`.

# Arguments
- `bs` can be either a `Tuple` or an `AbstractArray` of 0- and/or 1-dimensional bijectors
  - If `bs` is a `Tuple`, implementations are type-stable using generated functions
  - If `bs` is an `AbstractArray`, implementations are _not_ type-stable and use iterative methods
- `ranges` needs to be an iterable consisting of `UnitRange{Int}`
  - `length(bs) == length(ranges)` needs to be true.

# Examples
```
b1 = Logit(0.0, 1.0)
b2 = identity
b = stack(b1, b2)
b([0.0, 1.0]) == [b1(0.0), 1.0]  # => true
```
"""
struct Stacked{Bs,Rs<:Union{Tuple,AbstractArray}} <: Transform
    bs::Bs
    ranges_in::Rs
    ranges_out::Rs
    length_in::Int
    length_out::Int
end

function Stacked(bs::AbstractArray, ranges_in::AbstractArray)
    ranges_out = determine_output_ranges(bs, ranges_in)
    return Stacked{typeof(bs),typeof(ranges_in)}(
        bs, ranges_in, ranges_out, sum(length, ranges_in), sum(length, ranges_out)
    )
end
function Stacked(bs::Tuple, ranges_in::Tuple)
    ranges_out = determine_output_ranges(bs, ranges_in)
    return Stacked{typeof(bs),typeof(ranges_in)}(
        bs, ranges_in, ranges_out, sum(length, ranges_in), sum(length, ranges_out)
    )
end
Stacked(bs::AbstractArray, ranges::Tuple) = Stacked(bs, collect(ranges))
Stacked(bs::Tuple, ranges::AbstractArray) = Stacked(collect(bs), ranges)
Stacked(bs::Tuple) = Stacked(bs, ntuple(i -> i:i, length(bs)))
Stacked(bs::AbstractArray) = Stacked(bs, [i:i for i in 1:length(bs)])
Stacked(bs...) = Stacked(bs, ntuple(i -> i:i, length(bs)))

function determine_output_ranges(bs, ranges)
    offset = 0
    return map(bs, ranges) do b, r
        out_length = output_length(b, length(r))
        r = offset .+ (1:out_length)
        offset += out_length
        return r
    end
end

# NOTE: I don't like this but it seems necessary because `Stacked(...)` can occur in hot code paths.
function determine_output_ranges(bs::Tuple, ranges::Tuple)
    return determine_output_ranges_generated(bs, ranges)
end
@generated function determine_output_ranges_generated(bs::Tuple, ranges::Tuple)
    N = length(bs.parameters)
    exprs = []
    push!(exprs, :(offset = 0))

    rsyms = []
    for i in 1:N
        rsym = Symbol("r_$i")
        lengthsym = Symbol("length_$i")
        push!(exprs, :($lengthsym = output_length(bs[$i], length(ranges[$i]))))
        push!(exprs, :($rsym = offset .+ (1:($lengthsym))))
        push!(exprs, :(offset += $lengthsym))

        push!(rsyms, rsym)
    end

    acc_expr = Expr(:tuple, rsyms...)

    return quote
        $(exprs...)
        return $acc_expr
    end
end

# Avoid mixing tuples and arrays.
Stacked(bs::Tuple, ranges::AbstractArray) = Stacked(collect(bs), ranges)

Functors.@functor Stacked (bs,)

function Base.show(io::IO, b::Stacked)
    return print(io, "Stacked($(b.bs), $(b.ranges_in), $(b.ranges_out))")
end

function Base.:(==)(b1::Stacked, b2::Stacked)
    bs1, bs2 = b1.bs, b2.bs
    if !(bs1 isa Tuple && bs2 isa Tuple || bs1 isa Vector && bs2 isa Vector)
        return false
    end
    return all(bs1 .== bs2) &&
           all(b1.ranges_in .== b2.ranges_in) &&
           all(b1.ranges_out .== b2.ranges_out)
end

isclosedform(b::Stacked) = all(isclosedform, b.bs)

isinvertible(b::Stacked) = all(isinvertible, b.bs)

# For some reason `inverse.(sb.bs)` was unstable... This works though.
function inverse(sb::Stacked)
    return Stacked(
        map(inverse, sb.bs), sb.ranges_out, sb.ranges_in, sb.length_out, sb.length_in
    )
end
# map is not type stable for many stacked bijectors as a large tuple
# hence the generated function
@generated function inverse(sb::Stacked{A}) where {A<:Tuple}
    exprs = []
    for i in 1:length(A.parameters)
        push!(exprs, :(inverse(sb.bs[$i])))
    end
    return :(Stacked(
        ($(exprs...),), sb.ranges_out, sb.ranges_in, sb.length_out, sb.length_in
    ))
end

output_size(b::Stacked, sz::Tuple{Int}) = (b.length_out,)

@generated function _transform_stacked_recursive(
    x, rs::NTuple{N,UnitRange{Int}}, bs...
) where {N}
    exprs = []
    for i in 1:N
        push!(exprs, :(bs[$i](x[rs[$i]])))
    end
    return :(vcat($(exprs...)))
end
function _transform_stacked_recursive(x, rs::NTuple{1,UnitRange{Int}}, b)
    rs[1] == 1:length(x) || error("range must be 1:length(x)")
    return b(x)
end
function _transform_stacked(sb::Stacked{<:Tuple,<:Tuple}, x::AbstractVector{<:Real})
    y = _transform_stacked_recursive(x, sb.ranges_in, sb.bs...)
    return y
end
# The Stacked{<:AbstractArray} version is not TrackedArray friendly
function _transform_stacked(sb::Stacked{<:AbstractArray}, x::AbstractVector{<:Real})
    N = length(sb.bs)
    N == 1 && return sb.bs[1](x[sb.ranges_in[1]])

    y = mapvcat(1:N) do i
        sb.bs[i](x[sb.ranges_in[i]])
    end
    return y
end

function transform(sb::Stacked, x::AbstractVector{<:Real})
    if sb.length_in != length(x)
        error("input length mismatch ($(sb.length_in) != $(length(x)))")
    end
    y = _transform_stacked(sb, x)
    if sb.length_out != length(y)
        error("output length mismatch ($(sb.length_out) != $(length(y)))")
    end
    return y
end

function logabsdetjac(b::Stacked, x::AbstractVector{<:Real})
    N = length(b.bs)
    init = sum(logabsdetjac(b.bs[1], x[b.ranges_in[1]]))

    return if N > 1
        init + sum(2:N) do i
            sum(logabsdetjac(b.bs[i], x[b.ranges_in[i]]))
        end
    else
        init
    end
end

function logabsdetjac(
    b::Stacked{<:NTuple{N,Any},<:NTuple{N,Any}}, x::AbstractVector{<:Real}
) where {N}
    init = sum(logabsdetjac(b.bs[1], x[b.ranges_in[1]]))

    return if N == 1
        init
    else
        init + sum(2:N) do i
            sum(logabsdetjac(b.bs[i], x[b.ranges_in[i]]))
        end
    end
end

# Generates something similar to:
#
# quote
#     (y_1, _logjac) = with_logabsdet_jacobian(b.bs[1], x[b.ranges[1]])
#     logjac = sum(_logjac)
#     (y_2, _logjac) = with_logabsdet_jacobian(b.bs[2], x[b.ranges[2]])
#     logjac += sum(_logjac)
#     return (vcat(y_1, y_2), logjac)
# end
@generated function _with_logabsdet_jacobian(
    b::Stacked{<:NTuple{N,Any},<:NTuple{N,Any}}, x::AbstractVector
) where {N}
    expr = Expr(:block)
    y_names = []

    push!(
        expr.args, :((y_1, _logjac) = with_logabsdet_jacobian(b.bs[1], x[b.ranges_in[1]]))
    )
    # TODO: drop the `sum` when we have dimensionality
    push!(expr.args, :(logjac = sum(_logjac)))
    push!(y_names, :y_1)
    for i in 2:N
        y_name = Symbol("y_$i")
        push!(
            expr.args,
            :(($y_name, _logjac) = with_logabsdet_jacobian(b.bs[$i], x[b.ranges_in[$i]])),
        )

        # TODO: drop the `sum` when we have dimensionality
        push!(expr.args, :(logjac += sum(_logjac)))

        push!(y_names, y_name)
    end

    push!(expr.args, :(return (vcat($(y_names...)), logjac)))
    return expr
end

function _with_logabsdet_jacobian(sb::Stacked, x::AbstractVector)
    N = length(sb.bs)
    yinit, linit = with_logabsdet_jacobian(sb.bs[1], x[sb.ranges_in[1]])
    logjac = sum(linit)
    ys = mapreduce(vcat, sb.bs[2:end], sb.ranges_in[2:end]; init=yinit) do b, r
        y, l = with_logabsdet_jacobian(b, x[r])
        logjac += sum(l)
        y
    end
    return (ys, logjac)
end

function with_logabsdet_jacobian(sb::Stacked, x::AbstractVector)
    if sb.length_in != length(x)
        error("input length mismatch ($(sb.length_in) != $(length(x)))")
    end
    y, logjac = _with_logabsdet_jacobian(sb, x)
    if output_length(sb, length(x)) != length(y)
        error("output length mismatch ($(output_length(sb, length(x))) != $(length(y)))")
    end

    return (y, logjac)
end
