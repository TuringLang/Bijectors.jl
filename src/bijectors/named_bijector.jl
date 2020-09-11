abstract type AbstractNamedBijector <: AbstractBijector end

forward(b::AbstractBijector, x) = (rv = b(x), logabsdetjac = logabsdetjac(b, x))

#######################
### `NamedBijector` ###
#######################
struct NamedBijector{names, Bs<:NamedTuple{names}} <: AbstractNamedBijector
    bs::Bs
end

@inline names_to_bijectors(b::NamedBijector) = b.bs

@generated function (b::NamedBijector{names})(x::NamedTuple) where {names}
    return :(merge(x, ($([:($n = b.bs.$n(x.$n)) for n in names]...), )))
end

@generated function Base.inv(b::NamedBijector{names}) where {names}
    return :(NamedBijector(($([:($n = inv(b.bs.$n)) for n in names]...), )))
end

@generated function logabsdetjac(b::NamedBijector{names}, x::NamedTuple) where {names}
    return :(sum([$([:(logabsdetjac(b.bs.$n, x.$n)) for n in names]...), ]))
end


######################
### `NamedInverse` ###
######################
struct NamedInverse{B<:AbstractNamedBijector} <: AbstractNamedBijector
    orig::B
end
Base.inv(nb::AbstractNamedBijector) = NamedInverse(nb)
Base.inv(ni::NamedInverse) = ni.orig

logabsdetjac(ni::NamedInverse, y::NamedTuple) = -logabsdetjac(inv(ni), ni(y))

##########################
### `NamedComposition` ###
##########################
struct NamedComposition{Bs} <: AbstractNamedBijector
    bs::Bs
end

# Essentially just copy-paste from impl of composition for 'standard' bijectors,
# with minor changes here and there.
composel(bs::AbstractNamedBijector...) = NamedComposition(bs)
composer(bs::AbstractNamedBijector...) = NamedComposition(reverse(bs))
∘(b1::AbstractNamedBijector, b2::AbstractNamedBijector) = composel(b2, b1)

inv(ct::NamedComposition) = NamedComposition(reverse(map(inv, ct.bs)))

function (cb::NamedComposition{<:AbstractArray{<:AbstractNamedBijector}})(x)
    @assert length(cb.bs) > 0
    res = cb.bs[1](x)
    for b ∈ Base.Iterators.drop(cb.bs, 1)
        res = b(res)
    end

    return res
end

@generated function (cb::NamedComposition{T})(x) where {T<:Tuple}
    @assert length(T.parameters) > 0
    expr = :(x)
    for i in 1:length(T.parameters)
        expr = :(cb.bs[$i]($expr))
    end
    return expr
end

function logabsdetjac(cb::NamedComposition, x)
    y, logjac = forward(cb.bs[1], x)
    for i = 2:length(cb.bs)
        res = forward(cb.bs[i], y)
        y = res.rv
        logjac += res.logabsdetjac
    end

    return logjac
end

@generated function logabsdetjac(cb::NamedComposition{T}, x) where {T<:Tuple}
    N = length(T.parameters)

    expr = Expr(:block)
    push!(expr.args, :((y, logjac) = forward(cb.bs[1], x)))

    for i = 2:N - 1
        temp = gensym(:res)
        push!(expr.args, :($temp = forward(cb.bs[$i], y)))
        push!(expr.args, :(y = $temp.rv))
        push!(expr.args, :(logjac += $temp.logabsdetjac))
    end
    # don't need to evaluate the last bijector, only it's `logabsdetjac`
    push!(expr.args, :(logjac += logabsdetjac(cb.bs[$N], y)))

    push!(expr.args, :(return logjac))

    return expr
end


function forward(cb::NamedComposition, x)
    rv, logjac = forward(cb.bs[1], x)
    
    for t in cb.bs[2:end]
        res = forward(t, rv)
        rv = res.rv
        logjac = res.logabsdetjac + logjac
    end
    return (rv=rv, logabsdetjac=logjac)
end


@generated function forward(cb::NamedComposition{T}, x) where {T<:Tuple}
    expr = Expr(:block)
    push!(expr.args, :((y, logjac) = forward(cb.bs[1], x)))
    for i = 2:length(T.parameters)
        temp = gensym(:temp)
        push!(expr.args, :($temp = forward(cb.bs[$i], y)))
        push!(expr.args, :(y = $temp.rv))
        push!(expr.args, :(logjac += $temp.logabsdetjac))
    end
    push!(expr.args, :(return (rv = y, logabsdetjac = logjac)))

    return expr
end


############################
### `NamedCouplingLayer` ###
############################
struct NamedCoupling{target, deps, F} <: AbstractNamedBijector where {F, target}
    f::F
end

NamedCoupling(target, deps, f::F) where {F} = NamedCoupling{target, deps, F}(f)
function NamedCoupling(::Val{target}, ::Val{deps}, f::F) where {target, deps, F}
    return NamedCoupling{target, deps, F}(f)
end

coupling(b::NamedCoupling) = b.f
# For some reason trying to use the parameteric types doesn't always work
# so we have to do this weird approach of extracting type and then index `parameters`.
target(b::NamedCoupling) = typeof(b).parameters[1]
deps(b::NamedCoupling) = typeof(b).parameters[2]

@generated function (nc::NamedCoupling{target, deps, F})(x::NamedTuple) where {target, deps, F}
    return quote
        b = nc.f($([:(x.$d) for d in deps]...))
        return merge(x, ($target = b(x.$target), ))
    end
end

@generated function (ni::NamedInverse{<:NamedCoupling{target, deps, F}})(
    x::NamedTuple
) where {target, deps, F}
    return quote
        b = ni.orig.f($([:(x.$d) for d in deps]...))
        return merge(x, ($target = inv(b)(x.$target), ))
    end
end

@generated function logabsdetjac(nc::NamedCoupling{target, deps, F}, x::NamedTuple) where {target, deps, F}
    return quote
        b = nc.f($([:(x.$d) for d in deps]...))
        return logabsdetjac(b, x.$target)
    end
end
