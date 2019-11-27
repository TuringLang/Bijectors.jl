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
struct Composed{A, N} <: Bijector{N}
    ts::A

    Composed(bs::C) where {N, C<:Tuple{Vararg{<:Bijector{N}}}} = new{C, N}(bs)
    Composed(bs::A) where {N, A<:AbstractArray{<:Bijector{N}}} = new{A, N}(bs)
end

isclosedform(b::Composed) = all(isclosedform.(b.ts))


"""
    composel(ts::Bijector...)::Composed{<:Tuple}

Constructs `Composed` such that `ts` are applied left-to-right.
"""
composel(ts::Bijector{N}...) where {N} = Composed(ts)

"""
    composer(ts::Bijector...)::Composed{<:Tuple}

Constructs `Composed` such that `ts` are applied right-to-left.
"""
composer(ts::Bijector{N}...) where {N} = Composed(reverse(ts))

# The transformation of `Composed` applies functions left-to-right
# but in mathematics we usually go from right-to-left; this reversal ensures that
# when we use the mathematical composition ∘ we get the expected behavior.
# TODO: change behavior of `transform` of `Composed`?
@generated function ∘(b1::Bijector{N1}, b2::Bijector{N2}) where {N1, N2}
    if N1 == N2
        return :(composel(b2, b1))
    else
        return :(throw(DimensionMismatch("$(typeof(b1)) expects $(N1)-dim but $(typeof(b2)) expects $(N2)-dim")))
    end
end

# type-stable composition rules
∘(b1::Composed{<:Tuple}, b2::Bijector) = composel(b2, b1.ts...)
∘(b1::Bijector, b2::Composed{<:Tuple}) = composel(b2.ts..., b1)
∘(b1::Composed{<:Tuple}, b2::Composed{<:Tuple}) = composel(b2.ts..., b1.ts...)

# type-unstable composition rules
∘(b1::Composed{<:AbstractArray}, b2::Bijector) = Composed(pushfirst!(copy(b1.ts), b2))
∘(b1::Bijector, b2::Composed{<:AbstractArray}) = Composed(push!(copy(b2.ts), b1))
function ∘(b1::Composed{<:AbstractArray}, b2::Composed{<:AbstractArray})
    return Composed(append!(copy(b2.ts), copy(b1.ts)))
end


∘(::Identity{N}, ::Identity{N}) where {N} = Identity{N}()
∘(::Identity{N}, b::Bijector{N}) where {N} = b
∘(b::Bijector{N}, ::Identity{N}) where {N} = b

inv(ct::Composed) = Composed(reverse(map(inv, ct.ts)))
@generated function inv(cb::Composed{A}) where {A<:Tuple}
    exprs = []

    # inversion → reversing order
    for i = length(A.parameters):-1:1
        push!(exprs, :(inv(cb.ts[$i])))
    end
    return :(Composed(($(exprs...), )))
end

# # TODO: should arrays also be using recursive implementation instead?
function (cb::Composed{<:AbstractArray{<:Bijector}})(x)
    res = x
    for b ∈ cb.ts
        res = b(res)
    end

    return res
end

@generated function (cb::Composed{T})(x) where {T<:Tuple}
    expr = Expr(:block)
    push!(expr.args, :(y = cb.ts[1](x)))
    for i = 2:length(T.parameters)
        push!(expr.args, :(y = cb.ts[$i](y)))
    end
    push!(expr.args, :(return y))

    return expr
end

function logabsdetjac(cb::Composed, x)
    y, logjac = forward(cb.ts[1], x)
    for i = 2:length(cb.ts)
        res = forward(cb.ts[i], y)
        y = res.rv
        logjac += res.logabsdetjac
    end

    return (rv = y, logabsdetjac = logjac)
end

@generated function logabsdetjac(cb::Composed{T}, x) where {T<:Tuple}
    N = length(T.parameters)

    expr = Expr(:block)
    push!(expr.args, :((y, logjac) = forward(cb.ts[1], x)))

    for i = 2:N - 1
        push!(expr.args, :(res = forward(cb.ts[$i], y)))
        push!(expr.args, :(y = res.rv))
        push!(expr.args, :(logjac += res.logabsdetjac))
    end
    # don't need to evaluate the last bijector, only it's `logabsdetjac`
    push!(expr.args, :(logjac += logabsdetjac(cb.ts[$N], y)))

    push!(expr.args, :(return logjac))

    return expr
end


function forward(cb::Composed, x)
    rv, logjac = forward(cb.ts[1], x)
    
    for t in cb.ts[2:end]
        res = forward(t, rv)
        rv = res.rv
        logjac = res.logabsdetjac + logjac
    end
    return (rv=rv, logabsdetjac=logjac)
end


@generated function forward(cb::Composed{T}, x) where {T<:Tuple}
    expr = Expr(:block)
    push!(expr.args, :((y, logjac) = forward(cb.ts[1], x)))
    for i = 2:length(T.parameters)
        push!(expr.args, :(res = forward(cb.ts[$i], y)))
        push!(expr.args, :(y = res.rv))
        push!(expr.args, :(logjac += res.logabsdetjac))
    end
    push!(expr.args, :(return (rv = y, logabsdetjac = logjac)))

    return expr
end
