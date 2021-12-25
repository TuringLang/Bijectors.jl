abstract type AbstractNamedBijector <: Bijector end

forward(b::AbstractNamedBijector, x) = (result = b(x), logabsdetjac = logabsdetjac(b, x))

#######################
### `NamedBijector` ###
#######################
"""
    NamedBijector <: AbstractNamedBijector

Wraps a `NamedTuple` of key -> `Bijector` pairs, implementing evaluation, inversion, etc.

# Examples
```julia-repl
julia> using Bijectors: NamedBijector, Scale, Exp

julia> b = NamedBijector((a = Scale(2.0), b = Exp()));

julia> x = (a = 1., b = 0., c = 42.);

julia> b(x)
(a = 2.0, b = 1.0, c = 42.0)

julia> (a = 2 * x.a, b = exp(x.b), c = x.c)
(a = 2.0, b = 1.0, c = 42.0)
```
"""
struct NamedBijector{names, Bs<:NamedTuple{names}} <: AbstractNamedBijector
    bs::Bs
end

# fields contain nested numerical parameters
function Functors.functor(::Type{<:NamedBijector{names}}, x) where names
    function reconstruct_namedbijector(xs)
        return NamedBijector{names,typeof(xs.bs)}(xs.bs)
    end
    return (bs = x.bs,), reconstruct_namedbijector
end

names_to_bijectors(b::NamedBijector) = b.bs

@generated function (b::NamedBijector{names1})(
    x::NamedTuple{names2}
) where {names1, names2}
    exprs = []
    for n in names2
        if n in names1
            # Use processed value
            push!(exprs, :($n = b.bs.$n(x.$n)))
        else
            # Use existing value
            push!(exprs, :($n = x.$n))
        end
    end
    return :($(exprs...), )
end

@generated function inverse(b::NamedBijector{names}) where {names}
    return :(NamedBijector(($([:($n = inverse(b.bs.$n)) for n in names]...), )))
end

@generated function logabsdetjac(b::NamedBijector{names}, x::NamedTuple) where {names}
    exprs = [:(logabsdetjac(b.bs.$n, x.$n)) for n in names]
    return :(+($(exprs...)))
end

############################
### `NamedCouplingLayer` ###
############################
# TODO: Add ref to `Coupling` or `CouplingLayer` once that's merged.
"""
    NamedCoupling{target, deps, F} <: AbstractNamedBijector

Implements a coupling layer for named bijectors.

# Examples
```julia-repl
julia> using Bijectors: NamedCoupling, Scale

julia> b = NamedCoupling(:b, (:a, :c), (a, c) -> Scale(a + c))
NamedCoupling{:b,(:a, :c),var"#3#4"}(var"#3#4"())

julia> x = (a = 1., b = 2., c = 3.);

julia> b(x)
(a = 1.0, b = 8.0, c = 3.0)

julia> (a = x.a, b = (x.a + x.c) * x.b, c = x.c)
(a = 1.0, b = 8.0, c = 3.0)
```
"""
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
target(b::NamedCoupling{Target}) where {Target} = Target
deps(b::NamedCoupling{<:Any, Deps}) where {Deps} = Deps

@generated function (nc::NamedCoupling{target, deps, F})(x::NamedTuple) where {target, deps, F}
    return quote
        b = nc.f($([:(x.$d) for d in deps]...))
        return merge(x, ($target = b(x.$target), ))
    end
end

@generated function (ni::Inverse{<:NamedCoupling{target, deps, F}})(
    x::NamedTuple
) where {target, deps, F}
    return quote
        b = ni.orig.f($([:(x.$d) for d in deps]...))
        return merge(x, ($target = inverse(b)(x.$target), ))
    end
end

@generated function logabsdetjac(nc::NamedCoupling{target, deps, F}, x::NamedTuple) where {target, deps, F}
    return quote
        b = nc.f($([:(x.$d) for d in deps]...))
        return logabsdetjac(b, x.$target)
    end
end
