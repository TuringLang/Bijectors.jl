###############
# Composition #
###############

"""
    Composed(ts::A)

    ∘(b1::Bijector{N}, b2::Bijector{N})::Composed{<:Tuple}
    composel(ts::Bijector{N}...)::Composed{<:Tuple}
    composer(ts::Bijector{N}...)::Composed{<:Tuple}

where `A` refers to either
- `Tuple{Vararg{<:Bijector{N}}}`: a tuple of bijectors of dimensionality `N`
- `AbstractArray{<:Bijector{N}}`: an array of bijectors of dimensionality `N`

A `Bijector` representing composition of bijectors. `composel` and `composer` results in a
`Composed` for which application occurs from left-to-right and right-to-left, respectively.

Note that all the alternative ways of constructing a `Composed` returns a `Tuple` of bijectors.
This ensures type-stability of implementations of all relating methdos, e.g. `inv`.

If you want to use an `Array` as the container instead you can do

    Composed([b1, b2, ...])

In general this is not advised since you lose type-stability, but there might be cases
where this is desired, e.g. if you have a insanely large number of bijectors to compose.

# Examples
## Simple example
Let's consider a simple example of `Exp`:
```julia-repl
julia> using Bijectors: Exp

julia> b = Exp()
Exp{0}()

julia> b ∘ b
Composed{Tuple{Exp{0},Exp{0}},0}((Exp{0}(), Exp{0}()))

julia> (b ∘ b)(1.0) == exp(exp(1.0))    # evaluation
true

julia> inv(b ∘ b)(exp(exp(1.0))) == 1.0 # inversion
true

julia> logabsdetjac(b ∘ b, 1.0)         # determinant of jacobian
3.718281828459045
```

# Notes
## Order
It's important to note that `∘` does what is expected mathematically, which means that the
bijectors are applied to the input right-to-left, e.g. first applying `b2` and then `b1`:
```julia
(b1 ∘ b2)(x) == b1(b2(x))     # => true
```
But in the `Composed` struct itself, we store the bijectors left-to-right, so that
```julia
cb1 = b1 ∘ b2                  # => Composed.ts == (b2, b1)
cb2 = composel(b2, b1)         # => Composed.ts == (b2, b1)
cb1(x) == cb2(x) == b1(b2(x))  # => true
```

## Structure
`∘` will result in "flatten" the composition structure while `composel` and
`composer` preserve the compositional structure. This is most easily seen by an example:
```julia-repl
julia> b = Exp()
Exp{0}()

julia> cb1 = b ∘ b; cb2 = b ∘ b;

julia> (cb1 ∘ cb2).ts # <= different
(Exp{0}(), Exp{0}(), Exp{0}(), Exp{0}())

julia> (cb1 ∘ cb2).ts isa NTuple{4, Exp{0}}
true

julia> Bijectors.composer(cb1, cb2).ts
(Composed{Tuple{Exp{0},Exp{0}},0}((Exp{0}(), Exp{0}())), Composed{Tuple{Exp{0},Exp{0}},0}((Exp{0}(), Exp{0}())))

julia> Bijectors.composer(cb1, cb2).ts isa Tuple{Composed, Composed}
true
```

"""
struct Composed{A} <: Transform
    ts::A
end

# field contains nested numerical parameters
Functors.@functor Composed

invertible(cb::Composed) = sum(map(invertible, cb.ts))

isclosedform(b::Composed) = all(isclosedform, b.ts)

function Base.:(==)(b1::Composed, b2::Composed)
    ts1, ts2 = b1.ts, b2.ts
    return length(ts1) == length(ts2) && all(x == y for (x, y) in zip(ts1, ts2))
end

"""
    composel(ts::Transform...)::Composed{<:Tuple}

Constructs `Composed` such that `ts` are applied left-to-right.
"""
composel(ts::Transform...) = Composed(ts)

"""
    composer(ts::Transform...)::Composed{<:Tuple}

Constructs `Composed` such that `ts` are applied right-to-left.
"""
composer(ts::Transform...) = Composed(reverse(ts))

# The transformation of `Composed` applies functions left-to-right
# but in mathematics we usually go from right-to-left; this reversal ensures that
# when we use the mathematical composition ∘ we get the expected behavior.
# TODO: change behavior of `transform` of `Composed`?
∘(b1::Transform, b2::Transform) = composel(b2, b1)

# type-stable composition rules
∘(b1::Composed{<:Tuple}, b2::Transform) = composel(b2, b1.ts...)
∘(b1::Transform, b2::Composed{<:Tuple}) = composel(b2.ts..., b1)
∘(b1::Composed{<:Tuple}, b2::Composed{<:Tuple}) = composel(b2.ts..., b1.ts...)

# type-unstable composition rules
∘(b1::Composed{<:AbstractArray}, b2::Transform) = Composed(pushfirst!(copy(b1.ts), b2))
∘(b1::Transform, b2::Composed{<:AbstractArray}) = Composed(push!(copy(b2.ts), b1))
function ∘(b1::Composed{<:AbstractArray}, b2::Composed{<:AbstractArray})
    return Composed(append!(copy(b2.ts), copy(b1.ts)))
end

# if combining type-unstable and type-stable, return type-unstable
function ∘(b1::T1, b2::T2) where {T1<:Composed{<:Tuple}, T2<:Composed{<:AbstractArray}}
    error("Cannot compose compositions of different container-types; ($T1, $T2)")
end
function ∘(b1::T1, b2::T2) where {T1<:Composed{<:AbstractArray}, T2<:Composed{<:Tuple}}
    error("Cannot compose compositions of different container-types; ($T1, $T2)")
end


∘(::Identity, ::Identity) = Identity()
∘(::Identity, b::Transform) = b
∘(b::Transform, ::Identity) = b

inv(ct::Composed, ::Invertible) = Composed(reverse(map(inv, ct.ts)))

# TODO: should arrays also be using recursive implementation instead?
function transform(cb::Composed, x)
    @assert length(cb.ts) > 0
    res = cb.ts[1](x)
    for b ∈ Base.Iterators.drop(cb.ts, 1)
        res = b(res)
    end

    return res
end

function transform_batch(cb::Composed, x)
    @assert length(cb.ts) > 0
    res = cb.ts[1].(x)
    for b ∈ Base.Iterators.drop(cb.ts, 1)
        res = b.(res)
    end

    return res
end

@generated function transform(cb::Composed{T}, x) where {T<:Tuple}
    @assert length(T.parameters) > 0
    expr = :(x)
    for i in 1:length(T.parameters)
        expr = :(cb.ts[$i]($expr))
    end
    return expr
end

@generated function transform_batch(cb::Composed{T}, x) where {T<:Tuple}
    @assert length(T.parameters) > 0
    expr = :(x)
    for i in 1:length(T.parameters)
        expr = :(transform_batch(cb.ts[$i], $expr))
    end
    return expr
end

function logabsdetjac(cb::Composed, x)
    y, logjac = forward(cb.ts[1], x)
    for i = 2:length(cb.ts)
        res = forward(cb.ts[i], y)
        y = res.result
        logjac += res.logabsdetjac
    end

    return logjac
end

function logabsdetjac_batch(cb::Composed, x)
    init = forward(cb.ts[1], x)
    result = reduce(cb.ts[2:end]; init = init) do (y, logjac), b
        return forward(b, y)
    end

    return result.logabsdetjac
end


@generated function logabsdetjac(cb::Composed{T}, x) where {T<:Tuple}
    N = length(T.parameters)

    expr = Expr(:block)
    push!(expr.args, :((y, logjac) = forward(cb.ts[1], x)))

    for i = 2:N - 1
        temp = gensym(:res)
        push!(expr.args, :($temp = forward(cb.ts[$i], y)))
        push!(expr.args, :(y = $temp.result))
        push!(expr.args, :(logjac += $temp.logabsdetjac))
    end
    # don't need to evaluate the last bijector, only it's `logabsdetjac`
    push!(expr.args, :(logjac += logabsdetjac(cb.ts[$N], y)))

    push!(expr.args, :(return logjac))

    return expr
end

"""
    logabsdetjac_batch(cb::Composed{<:Tuple}, x)

Generates something of the form
```julia
quote
    (y, logjac_1) = forward_batch(cb.ts[1], x)
    logjac_2 = logabsdetjac_batch(cb.ts[2], y)
    return logjac_1 + logjac_2
end
```
"""
@generated function logabsdetjac_batch(cb::Composed{T}, x) where {T<:Tuple}
    N = length(T.parameters)

    expr = Expr(:block)
    push!(expr.args, :((y, logjac_1) = forward_batch(cb.ts[1], x)))

    for i = 2:N - 1
        temp = gensym(:res)
        push!(expr.args, :($temp = forward_batch(cb.ts[$i], y)))
        push!(expr.args, :(y = $temp.result))
        push!(expr.args, :($(Symbol("logjac_$i")) = $temp.logabsdetjac))
    end
    # don't need to evaluate the last bijector, only it's `logabsdetjac`
    push!(expr.args, :($(Symbol("logjac_$N")) = logabsdetjac_batch(cb.ts[$N], y)))

    sum_expr = Expr(:call, :+, [Symbol("logjac_$i") for i = 1:N]...)
    push!(expr.args, :(return $(sum_expr)))

    return expr
end


function forward(cb::Composed, x)
    result, logjac = forward(cb.ts[1], x)
    
    for t in cb.ts[2:end]
        res = forward(t, result)
        result = res.result
        logjac = res.logabsdetjac + logjac
    end
    return (result=result, logabsdetjac=logjac)
end

function forward_batch(cb::Composed, x)
    result, logjac = forward_batch(cb.ts[1], x)

    for t in cb.ts[2:end]
        res = forward_batch(t, result)
        result = res.result
        logjac = res.logabsdetjac + logjac
    end
    return (result=result, logabsdetjac=logjac)
end


@generated function forward(cb::Composed{T}, x) where {T<:Tuple}
    expr = Expr(:block)
    push!(expr.args, :((y, logjac) = forward(cb.ts[1], x)))
    for i = 2:length(T.parameters)
        temp = gensym(:temp)
        push!(expr.args, :($temp = forward(cb.ts[$i], y)))
        push!(expr.args, :(y = $temp.result))
        push!(expr.args, :(logjac += $temp.logabsdetjac))
    end
    push!(expr.args, :(return (result = y, logabsdetjac = logjac)))

    return expr
end

@generated function forward_batch(cb::Composed{T}, x) where {T<:Tuple}
    N = length(T.parameters)
    expr = Expr(:block)
    push!(expr.args, :((y, logjac_1) = forward_batch(cb.ts[1], x)))
    for i = 2:N
        temp = gensym(:temp)
        push!(expr.args, :($temp = forward_batch(cb.ts[$i], y)))
        push!(expr.args, :(y = $temp.result))
        push!(expr.args, :($(Symbol("logjac_$i")) = $temp.logabsdetjac))
    end

    sum_expr = Expr(:call, :+, [Symbol("logjac_$i") for i = 1:N]...)
    push!(expr.args, :(return (result = y, logabsdetjac = $(sum_expr))))

    return expr
end
