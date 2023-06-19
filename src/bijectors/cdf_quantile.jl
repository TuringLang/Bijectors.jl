"""
    CDFBijector(dist::Distributions.ContinuousUnivariateDistribution)

A [`Bijector`](@ref) that transforms the input from the support of the given distribution to
the unit interval using the cumulative distribution function of the distribution.

The inverse is [`QuantileBijector`](@ref).

# Example

```jldoctest
julia> using Bijectors: CDFBijector

julia> using Distributions: Normal

julia> b = CDFBijector(Normal());

julia> p = [0.1, 0.5, 0.9];

julia> transform(b, quantile.(Normal(), p)) ≈ p
true
```
"""
struct CDFBijector{D<:ContinuousUnivariateDistribution} <: Bijector
    dist::D
end

Base.:(==)(b1::CDFBijector, b2::CDFBijector) = b1.dist == b2.dist

Functors.@functor CDFBijector

function Base.show(io::IO, b::CDFBijector)
    print(io, "CDFBijector(")
    print(io, b.dist)
    print(io, ")")
    return nothing
end

with_logabsdet_jacobian(b::CDFBijector, x) = transform(b, x), logabsdetjac(b, x)

transform(b::CDFBijector, x) = Distributions.cdf.(b.dist, x)

logabsdetjac(b::CDFBijector, x) = Distributions.logpdf.(b.dist, x)


"""
    QuantileBijector(dist::Distributions.ContinuousUnivariateDistribution)

A [`Bijector`](@ref) that transforms the input from the unit interval to the support of the
given distribution using the quantile function of the distribution.

The inverse is [`CDFBijector`](@ref).

# Example

```jldoctest
julia> using Bijectors: QuantileBijector

julia> using Distributions: Gamma

julia> b = QuantileBijector(Gamma());

julia> p = [0.1, 0.5, 0.9];

julia> transform(b, p) ≈ quantile.(Gamma(), p)
true
```
"""
struct QuantileBijector{D<:ContinuousUnivariateDistribution} <: Bijector
    dist::D
end

Base.:(==)(b1::QuantileBijector, b2::QuantileBijector) = b1.dist == b2.dist

Functors.@functor QuantileBijector

function Base.show(io::IO, b::QuantileBijector)
    print(io, "QuantileBijector(")
    print(io, b.dist)
    print(io, ")")
    return nothing
end

with_logabsdet_jacobian(b::QuantileBijector, x) = transform(b, x), logabsdetjac(b, x)

transform(b::QuantileBijector, x) = Distributions.quantile.(b.dist, x)

logabsdetjac(b::QuantileBijector, x) = @. -Distributions.logpdf(b.dist, x)

inverse(b::CDFBijector) = QuantileBijector(b.dist)
inverse(b::QuantileBijector) = CDFBijector(b.dist)
