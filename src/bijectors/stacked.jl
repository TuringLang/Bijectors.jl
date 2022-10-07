"""
    Stacked(bs)
    Stacked(bs, ranges)
    stack(bs::Bijector...) # where `0` means 0-dim `Bijector`

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
b2 = Identity{0}()
b = stack(b1, b2)
b([0.0, 1.0]) == [b1(0.0), 1.0]  # => true
```
"""
struct Stacked{Bs, Rs} <: Transform
    bs::Bs
    ranges::Rs
end
Stacked(bs::Tuple) = Stacked(bs, ntuple(i -> i:i, length(bs)))
Stacked(bs::AbstractArray) = Stacked(bs, [i:i for i in 1:length(bs)])

# Avoid mixing tuples and arrays.
Stacked(bs::Tuple, ranges::AbstractArray) = Stacked(collect(bs), ranges)

Functors.@functor Stacked (bs,)

function Base.:(==)(b1::Stacked, b2::Stacked)
    bs1, bs2 = b1.bs, b2.bs
    if !(bs1 isa Tuple && bs2 isa Tuple || bs1 isa Vector && bs2 isa Vector)
        return false
    end
    return all(bs1 .== bs2) && all(b1.ranges .== b2.ranges)
end

isclosedform(b::Stacked) = all(isclosedform, b.bs)

invertible(b::Stacked) = sum(map(invertible, b.bs))

stack(bs...) = Stacked(bs)

# For some reason `inverse.(sb.bs)` was unstable... This works though.
inverse(sb::Stacked) = Stacked(map(inverse, sb.bs), sb.ranges)
# map is not type stable for many stacked bijectors as a large tuple
# hence the generated function
@generated function inverse(sb::Stacked{A}) where {A <: Tuple}
    exprs = []
    for i = 1:length(A.parameters)
        push!(exprs, :(inverse(sb.bs[$i])))
    end
    :(Stacked(($(exprs...), ), sb.ranges))
end

@generated function _transform(x, rs::NTuple{N, UnitRange{Int}}, bs...) where N
    exprs = []
    for i = 1:N
        push!(exprs, :(bs[$i](x[rs[$i]])))
    end
    return :(vcat($(exprs...)))
end
function _transform(x, rs::NTuple{1, UnitRange{Int}}, b)
    @assert rs[1] == 1:length(x)
    return b(x)
end
function transform(sb::Stacked{<:Tuple,<:Tuple}, x::AbstractVector{<:Real})
    y = _transform(x, sb.ranges, sb.bs...)
    @assert size(y) == size(x) "x is size $(size(x)) but y is $(size(y))"
    return y
end
# The Stacked{<:AbstractArray} version is not TrackedArray friendly
function transform(sb::Stacked{<:AbstractArray}, x::AbstractVector{<:Real})
    N = length(sb.bs)
    N == 1 && return sb.bs[1](x[sb.ranges[1]])

    y = mapvcat(1:N) do i
        sb.bs[i](x[sb.ranges[i]])
    end
    @assert size(y) == size(x) "x is size $(size(x)) but y is $(size(y))"
    return y
end

function logabsdetjac(
    b::Stacked,
    x::AbstractVector{<:Real}
)
    N = length(b.bs)
    init = sum(logabsdetjac(b.bs[1], x[b.ranges[1]]))

    return if N > 1
        init + sum(2:N) do i
            sum(logabsdetjac(b.bs[i], x[b.ranges[i]]))
        end
    else
        init
    end
end

function logabsdetjac(
    b::Stacked{<:Tuple{Vararg{<:Any, N}}, <:Tuple{Vararg{<:Any, N}}},
    x::AbstractVector{<:Real}
) where {N}
    init = sum(logabsdetjac(b.bs[1], x[b.ranges[1]]))

    return if N == 1
        init
    else
        init + sum(2:N) do i
            sum(logabsdetjac(b.bs[i], x[b.ranges[i]]))
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
@generated function with_logabsdet_jacobian(b::Stacked{<:Tuple{Vararg{<:Any, N}}, <:Tuple{Vararg{<:Any, N}}}, x::AbstractVector) where {N}
    expr = Expr(:block)
    y_names = []

    push!(expr.args, :((y_1, _logjac) = with_logabsdet_jacobian(b.bs[1], x[b.ranges[1]])))
    # TODO: drop the `sum` when we have dimensionality
    push!(expr.args, :(logjac = sum(_logjac)))
    push!(y_names, :y_1)
    for i = 2:N
        y_name = Symbol("y_$i")
        push!(expr.args, :(($y_name, _logjac) = with_logabsdet_jacobian(b.bs[$i], x[b.ranges[$i]])))

        # TODO: drop the `sum` when we have dimensionality
        push!(expr.args, :(logjac += sum(_logjac)))

        push!(y_names, y_name)
    end

    push!(expr.args, :(return (vcat($(y_names...)), logjac)))
    return expr
end

function with_logabsdet_jacobian(sb::Stacked, x::AbstractVector)
    N = length(sb.bs)
    yinit, linit = with_logabsdet_jacobian(sb.bs[1], x[sb.ranges[1]])
    logjac = sum(linit)
    ys = mapreduce(vcat, sb.bs[2:end], sb.ranges[2:end]; init=yinit) do b, r
        y, l = with_logabsdet_jacobian(b, x[r])
        logjac += sum(l)
        y
    end
    return (ys, logjac)
end
