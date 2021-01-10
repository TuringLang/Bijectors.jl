"""
    Stacked(bs)
    Stacked(bs, ranges)
    stack(bs::Bijector{0}...) # where `0` means 0-dim `Bijector`

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
struct Stacked{Bs, N} <: Bijector{1}
    bs::Bs
    ranges::NTuple{N, UnitRange{Int}}

    function Stacked(
        bs::C,
        ranges::NTuple{N, UnitRange{Int}}
    ) where {N, C<:Tuple{Vararg{<:ZeroOrOneDimBijector, N}}}
        return new{C, N}(bs, ranges)
    end

    function Stacked(
        bs::A,
        ranges::NTuple{N, UnitRange{Int}}
    ) where {N, A<:AbstractArray{<:Bijector}}
        @assert length(bs) == N "number of bijectors is not same as number of ranges"
        @assert all(b -> isa(b, ZeroOrOneDimBijector), bs)
        return new{A, N}(bs, ranges)
    end
end
Stacked(bs, ranges::AbstractArray) = Stacked(bs, tuple(ranges...))
Stacked(bs) = Stacked(bs, tuple([i:i for i = 1:length(bs)]...))

# define nested numerical parameters
# TODO: replace with `Functors.@functor Stacked (bs,)` when
# https://github.com/FluxML/Functors.jl/pull/7 is merged
function Functors.functor(::Type{<:Stacked}, x)
    function reconstruct_stacked(xs)
        return Stacked(xs.bs, x.ranges)
    end
    return (bs = x.bs,), reconstruct_stacked
end

function Base.:(==)(b1::Stacked, b2::Stacked)
    bs1, bs2 = b1.bs, b2.bs
    if !(bs1 isa Tuple && bs2 isa Tuple || bs1 isa Vector && bs2 isa Vector)
        return false
    end
    return all(bs1 .== bs2) && all(b1.ranges .== b2.ranges)
end

isclosedform(b::Stacked) = all(isclosedform, b.bs)

stack(bs::Bijector{0}...) = Stacked(bs)

# For some reason `inv.(sb.bs)` was unstable... This works though.
inv(sb::Stacked) = Stacked(map(inv, sb.bs), sb.ranges)
# map is not type stable for many stacked bijectors as a large tuple
# hence the generated function
@generated function inv(sb::Stacked{A}) where {A <: Tuple}
    exprs = []
    for i = 1:length(A.parameters)
        push!(exprs, :(inv(sb.bs[$i])))
    end
    :(Stacked(($(exprs...), ), sb.ranges))
end

@generated function _transform(x, rs::NTuple{N, UnitRange{Int}}, bs::Bijector...) where N
    exprs = []
    for i = 1:N
        push!(exprs, :(bs[$i](x[rs[$i]])))
    end
    return :(vcat($(exprs...)))
end
function _transform(x, rs::NTuple{1, UnitRange{Int}}, b::Bijector)
    @assert rs[1] == 1:length(x)
    return b(x)
end
function (sb::Stacked{<:Tuple})(x::AbstractVector{<:Real})
    y = _transform(x, sb.ranges, sb.bs...)
    @assert size(y) == size(x) "x is size $(size(x)) but y is $(size(y))"
    return y
end
# The Stacked{<:AbstractArray} version is not TrackedArray friendly
function (sb::Stacked{<:AbstractArray, N})(x::AbstractVector{<:Real}) where {N}
    y = mapvcat(1:N) do i
        sb.bs[i](x[sb.ranges[i]])
    end
    @assert size(y) == size(x) "x is size $(size(x)) but y is $(size(y))"
    return y
end

(sb::Stacked)(x::AbstractMatrix{<:Real}) = eachcolmaphcat(sb, x)
function logabsdetjac(
    b::Stacked{<:Any, N},
    x::AbstractVector{<:Real}
) where {N}
    init = sum(logabsdetjac(b.bs[1], x[b.ranges[1]]))
    init + sum(2:N) do i
        sum(logabsdetjac(b.bs[i], x[b.ranges[i]]))
    end
end

function logabsdetjac(b::Stacked{<:Any, 1}, x::AbstractVector{<:Real})
    return sum(logabsdetjac(b.bs[1], x[b.ranges[1]]))
end

function logabsdetjac(b::Stacked, x::AbstractMatrix{<:Real})
    return map(eachcol(x)) do c
        logabsdetjac(b, c)
    end
end

# Generates something similar to:
#
# quote
#     (y_1, _logjac) = forward(b.bs[1], x[b.ranges[1]])
#     logjac = sum(_logjac)
#     (y_2, _logjac) = forward(b.bs[2], x[b.ranges[2]])
#     logjac += sum(_logjac)
#     return (rv = vcat(y_1, y_2), logabsdetjac = logjac)
# end
@generated function forward(b::Stacked{T, N}, x::AbstractVector) where {N, T<:Tuple}
    expr = Expr(:block)
    y_names = []

    push!(expr.args, :((y_1, _logjac) = forward(b.bs[1], x[b.ranges[1]])))
    # TODO: drop the `sum` when we have dimensionality
    push!(expr.args, :(logjac = sum(_logjac)))
    push!(y_names, :y_1)
    for i = 2:length(T.parameters)
        y_name = Symbol("y_$i")
        push!(expr.args, :(($y_name, _logjac) = forward(b.bs[$i], x[b.ranges[$i]])))

        # TODO: drop the `sum` when we have dimensionality
        push!(expr.args, :(logjac += sum(_logjac)))

        push!(y_names, y_name)
    end

    push!(expr.args, :(return (rv = vcat($(y_names...)), logabsdetjac = logjac)))
    return expr
end

function forward(sb::Stacked{<:AbstractArray, N}, x::AbstractVector) where {N}
    yinit, linit = forward(sb.bs[1], x[sb.ranges[1]])
    logjac = sum(linit)
    ys = mapvcat(drop(sb.bs, 1), drop(sb.ranges, 1)) do b, r
        y, l = forward(b, x[r])
        logjac += sum(l)
        y
    end
    return (rv = vcat(yinit, ys), logabsdetjac = logjac)
end
