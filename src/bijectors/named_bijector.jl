abstract type AbstractNamedTransform <: Transform end

#######################
### `NamedTransform` ###
#######################
"""
    NamedTransform <: AbstractNamedTransform

Wraps a `NamedTuple` of key -> `Bijector` pairs, implementing evaluation, inversion, etc.

# Examples
```jldoctest
julia> using Bijectors: NamedTransform, Scale

julia> b = NamedTransform((a = Scale(2.0), b = exp));

julia> x = (a = 1., b = 0., c = 42.);

julia> b(x)
(a = 2.0, b = 1.0, c = 42.0)

julia> (a = 2 * x.a, b = exp(x.b), c = x.c)
(a = 2.0, b = 1.0, c = 42.0)
```
"""
struct NamedTransform{names, Bs<:NamedTuple{names}} <: AbstractNamedTransform
    bs::Bs
end

# fields contain nested numerical parameters
function Functors.functor(::Type{<:NamedTransform{names}}, x) where names
    function reconstruct_namedbijector(xs)
        return NamedTransform{names,typeof(xs.bs)}(xs.bs)
    end
    return (bs = x.bs,), reconstruct_namedbijector
end

# TODO: Use recursion instead of `@generated`?
@generated function inverse(b::NamedTransform{names}) where {names}
    return :(NamedTransform(($([:($n = inverse(b.bs.$n)) for n in names]...), )))
end

@generated function transform(
    b::NamedTransform{names1},
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

@generated function logabsdetjac(b::NamedTransform{names}, x::NamedTuple) where {names}
    exprs = [:(logabsdetjac(b.bs.$n, x.$n)) for n in names]
    return :(+($(exprs...)))
end

@generated function with_logabsdet_jacobian(
    b::NamedTransform{names1},
    x::NamedTuple{names2}
) where {names1, names2}
    body_exprs = []
    logjac_expr = Expr(:call, :+)
    val_expr = Expr(:tuple, )
    for n in names2
        if n in names1
            val_sym = Symbol("y_$n")
            logjac_sym = Symbol("logjac_$n")

            push!(body_exprs, :(($val_sym, $logjac_sym) = with_logabsdet_jacobian(b.bs.$n, x.$n)))
            push!(logjac_expr.args, logjac_sym)
            push!(val_expr.args, :($n = $val_sym))
        else
            push!(val_expr.args, :($n = x.$n))
        end
    end
    return quote
        $(body_exprs...)
        return NamedTuple{$names2}($val_expr), $logjac_expr
    end
end

############################
### `NamedCouplingLayer` ###
############################
# TODO: Add ref to `Coupling` or `CouplingLayer` once that's merged.
"""
    NamedCoupling{target, deps, F} <: AbstractNamedTransform

Implements a coupling layer for named bijectors.

# Examples
```jldoctest
julia> using Bijectors: NamedCoupling, Scale

julia> b = NamedCoupling(:b, (:a, :c), (a, c) -> Scale(a + c));

julia> x = (a = 1., b = 2., c = 3.);

julia> b(x)
(a = 1.0, b = 8.0, c = 3.0)

julia> (a = x.a, b = (x.a + x.c) * x.b, c = x.c)
(a = 1.0, b = 8.0, c = 3.0)
```
"""
struct NamedCoupling{target, deps, F} <: AbstractNamedTransform where {F, target}
    f::F
end

NamedCoupling(target, deps, f::F) where {F} = NamedCoupling{target, deps, F}(f)
function NamedCoupling(::Val{target}, ::Val{deps}, f::F) where {target, deps, F}
    return NamedCoupling{target, deps, F}(f)
end

invertible(::NamedCoupling) = Invertible()

coupling(b::NamedCoupling) = b.f
# For some reason trying to use the parameteric types doesn't always work
# so we have to do this weird approach of extracting type and then index `parameters`.
target(b::NamedCoupling{Target}) where {Target} = Target
deps(b::NamedCoupling{<:Any, Deps}) where {Deps} = Deps

@generated function with_logabsdet_jacobian(nc::NamedCoupling{target, deps, F}, x::NamedTuple) where {target, deps, F}
    return quote
        b = nc.f($([:(x.$d) for d in deps]...))
        x_target, logjac = with_logabsdet_jacobian(b, x.$target)
        return merge(x, ($target = x_target, )), logjac
    end
end

@generated function with_logabsdet_jacobian(ni::Inverse{<:NamedCoupling{target, deps, F}}, x::NamedTuple) where {target, deps, F}
    return quote
        ib = inverse(ni.orig.f($([:(x.$d) for d in deps]...)))
        x_target, logjac = with_logabsdet_jacobian(ib, x.$target)
        return merge(x, ($target = x_target, )), logjac
    end
end
