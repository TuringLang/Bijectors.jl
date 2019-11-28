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
struct Stacked{Bs, N} <: Bijector{1} where N
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
        @assert all(isa.(bs, ZeroOrOneDimBijector))
        return new{A, N}(bs, ranges)
    end
end
Stacked(bs, ranges::AbstractArray) = Stacked(bs, tuple(ranges...))
Stacked(bs) = Stacked(bs, tuple([i:i for i = 1:length(bs)]...))

isclosedform(b::Stacked) = all(isclosedform.(b.bs))

stack(bs::Bijector{0}...) = Stacked(bs)

# For some reason `inv.(sb.bs)` was unstable... This works though.
inv(sb::Stacked) = Stacked(map(inv, sb.bs), sb.ranges)
@generated function inv(sb::Stacked{A}) where {A <: Tuple}
    exprs = []
    for i = 1:length(A.parameters)
        push!(exprs, :(inv(sb.bs[$i])))
    end
    :(Stacked(($(exprs...), ), sb.ranges))
end

# TODO: Is there a better approach to this?
@generated function _transform(x, rs::NTuple{N, UnitRange{Int}}, bs::Bijector...) where N
    exprs = []
    for i = 1:N
        push!(exprs, :(bs[$i](x[rs[$i]])))
    end

    return :(vcat($(exprs...)))
end
_transform(x, rs::NTuple{1, UnitRange{Int}}, b::Bijector) = b(x)

function (sb::Stacked{<:Tuple})(x::AbstractVector{<:Real})
    y = _transform(x, sb.ranges, sb.bs...)
    @assert size(y) == size(x) "x is size $(size(x)) but y is $(size(y))"
    return y
end
function (sb::Stacked{<:AbstractArray, N})(x::AbstractVector{<:Real}) where {N}
    y = vcat([sb.bs[i](x[sb.ranges[i]]) for i = 1:N]...)
    @assert size(y) == size(x) "x is size $(size(x)) but y is $(size(y))"
    return y
end

# HACK: `reshape` to get around the fact that `hcat` isn't type-stable
(sb::Stacked)(x::AbstractMatrix{<:Real}) = reshape(foldl(hcat, [sb(x[:, i]) for i = 1:size(x, 2)]), size(x)...)

# TODO: implement custom adjoint since we can exploit block-diagonal nature of `Stacked`
function (sb::Stacked)(x::TrackedArray{A, 2}) where {A}
    return Tracker.collect(hcat([sb(x[:, i]) for i = 1:size(x, 2)]...))
end

@generated function logabsdetjac(
    b::Stacked{<:Tuple, N},
    x::AbstractVector{<:Real}
) where {N}
    exprs = []
    for i = 1:N
        push!(exprs, :(sum(logabsdetjac(b.bs[$i], x[b.ranges[$i]]))))
    end

    return :(sum([$(exprs...), ]))
end
function logabsdetjac(
    b::Stacked{<:AbstractArray, N},
    x::AbstractVector{<:Real}
) where {N}
    # TODO: drop the `sum` when we have dimensionality
    return sum([sum(logabsdetjac(b.bs[i], x[b.ranges[i]])) for i = 1:N])
end
function logabsdetjac(b::Stacked, x::AbstractMatrix{<:Real})
    return [logabsdetjac(b, x[:, i]) for i = 1:size(x, 2)]
end
function logabsdetjac(b::Stacked, x::TrackedArray{A, 2}) where {A}
    return Tracker.collect([logabsdetjac(b, x[:, i]) for i = 1:size(x, 2)])
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
    ys = []
    logjacs = []
    for i = 1:N
        y, logjac = forward(sb.bs[i], x[sb.ranges[i]])
        push!(ys, y)
        # TODO: drop the `sum` when we have dimensionality
        push!(logjacs, sum(logjac))
    end

    return (rv = vcat(ys...), logabsdetjac = sum(logjacs))
end
