abstract type AbstractNamedBijector <: AbstractBijector end

struct NamedBijector{names, Bs<:NamedTuple{names}} <: AbstractNamedBijector
    bs::Bs
end

@inline names_to_bijectors(b::NamedBijector) = b.bs

@generated function (b::NamedBijector{names})(x::NamedTuple{names}) where {names}
    return :(($([:($n = b.bs.$n(x.$n)) for n in names]...), ))
end

@generated function Base.inv(b::NamedBijector{names}) where {names}
    return :(NamedBijector(($([:($n = inv(b.bs.$n)) for n in names]...), )))
end

@generated function logabsdetjac(b::NamedBijector{names}, x::NamedTuple{names}) where {names}
    return :(sum([$([:(logabsdetjac(b.bs.$n, x.$n)) for n in names]...), ]))
end

# FIXME: figure out how to do compositions between, e.g. `NamedBijector` and `NamedCoupling`
# Coouuld just do the same as we've done for `Bijector`, but it's a bit annoying and seems like a bit too much?
@generated function ∘(
        b1::NamedBijector{names}, 
        b2::NamedBijector{names}
) where {names}
    return :(NamedBijector(($([:($n = b1.bs.$n ∘ b2.bs.$n) for n in names]...), )))
end

# TODO: figure out a good way to do `composel` using generated functions?
function composel(bs::NamedBijector{names}...) where {names}
    return NamedBijector((; ((n, composel([b.bs[n] for b in bs]...)) for n in names)...))
end

function composer(bs::NamedBijector{names}...) where {names}
    return NamedBijector((; ((n, composer([b.bs[n] for b in bs]...)) for n in names)...))
end


####################
### NamedInverse ###
####################
struct NamedInverse{B<:AbstractNamedBijector} <: AbstractNamedBijector
    orig::B
end
Base.inv(nb::AbstractNamedBijector) = NamedInverse(nb)
Base.inv(ni::NamedInverse) = ni.orig

logabsdetjac(ni::NamedInverse, y::NamedTuple) = -logabsdetjac(inv(ni), ni(y))

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
