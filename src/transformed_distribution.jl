# Transformed distributions
struct TransformedDistribution{D, B, V} <: Distribution{V, Continuous} where {D<:Distribution{V, Continuous}, B<:Bijector}
    dist::D
    transform::B

    TransformedDistribution(d::UnivariateDistribution, b::Bijector{0}) = new{typeof(d), typeof(b), Univariate}(d, b)
    TransformedDistribution(d::MultivariateDistribution, b::Bijector{1}) = new{typeof(d), typeof(b), Multivariate}(d, b)
    TransformedDistribution(d::MatrixDistribution, b::Bijector{2}) = new{typeof(d), typeof(b), Matrixvariate}(d, b)
end

# fields may contain nested numerical parameters
Functors.@functor TransformedDistribution

const UnivariateTransformed = TransformedDistribution{<:Distribution, <:Bijector, Univariate}
const MultivariateTransformed = TransformedDistribution{<:Distribution, <:Bijector, Multivariate}
const MvTransformed = MultivariateTransformed
const MatrixTransformed = TransformedDistribution{<:Distribution, <:Bijector, Matrixvariate}
const Transformed = TransformedDistribution


"""
    transformed(d::Distribution)
    transformed(d::Distribution, b::Bijector)

Couples distribution `d` with the bijector `b` by returning a `TransformedDistribution`.

If no bijector is provided, i.e. `transformed(d)` is called, then 
`transformed(d, bijector(d))` is returned.
"""
transformed(d::Distribution, b::Bijector) = TransformedDistribution(d, b)
transformed(d) = transformed(d, bijector(d))

"""
    bijector(d::Distribution)

Returns the constrained-to-unconstrained bijector for distribution `d`.
"""
bijector(d::DiscreteUnivariateDistribution) = Identity{0}()
bijector(d::DiscreteMultivariateDistribution) = Identity{1}()
bijector(d::ContinuousUnivariateDistribution) = TruncatedBijector(minimum(d), maximum(d))
bijector(d::Product{Discrete}) = Identity{1}()
function bijector(d::Product{Continuous})
    return TruncatedBijector{1}(_minmax(d.v)...)
end
@generated function _minmax(d::AbstractArray{T}) where {T}
    try
        min, max = minimum(T), maximum(T)
        return :($min, $max)
    catch
        return :(minimum.(d), maximum.(d))
    end
end

bijector(d::Normal) = Identity{0}()
bijector(d::Distributions.AbstractMvNormal) = Identity{1}()
bijector(d::Distributions.AbstractMvLogNormal) = Log{1}()
bijector(d::PositiveDistribution) = Log{0}()
bijector(d::SimplexDistribution) = SimplexBijector{1}()
bijector(d::KSOneSided) = Logit(zero(eltype(d)), one(eltype(d)))

bijector_bounded(d, a=minimum(d), b=maximum(d)) = Logit(a, b)
bijector_lowerbounded(d, a=minimum(d)) = Log() ∘ Shift(-a)
bijector_upperbounded(d, b=maximum(d)) = Log() ∘ Shift(b) ∘ Scale(- one(typeof(b)))

const BoundedDistribution = Union{
    Arcsine, Biweight, Cosine, Epanechnikov, Beta, NoncentralBeta
}
bijector(d::BoundedDistribution) = bijector_bounded(d)

const LowerboundedDistribution = Union{Pareto, Levy}
bijector(d::LowerboundedDistribution) = bijector_lowerbounded(d)

bijector(d::PDMatDistribution) = PDBijector()
bijector(d::MatrixBeta) = PDBijector()

bijector(d::LKJ) = CorrBijector()

##############################
# Distributions.jl interface #
##############################

# size
Base.length(td::Transformed) = length(td.dist)
Base.size(td::Transformed) = size(td.dist)

function logpdf(td::UnivariateTransformed, y::Real)
    res = forward(inv(td.transform), y)
    return logpdf(td.dist, res.rv) + res.logabsdetjac
end

# TODO: implement more efficiently for flows in the case of `Matrix`
function logpdf(td::MvTransformed, y::AbstractMatrix{<:Real})
    # batch-implementation for multivariate
    res = forward(inv(td.transform), y)
    return logpdf(td.dist, res.rv) + res.logabsdetjac
end

function logpdf(td::MvTransformed{<:Dirichlet}, y::AbstractMatrix{<:Real})
    T = eltype(y)
    ϵ = _eps(T)

    res = forward(inv(td.transform), y)
    return logpdf(td.dist, mappedarray(x->x+ϵ, res.rv)) + res.logabsdetjac
end

function _logpdf(td::MvTransformed, y::AbstractVector{<:Real})
    res = forward(inv(td.transform), y)
    return logpdf(td.dist, res.rv) + res.logabsdetjac
end

function _logpdf(td::MvTransformed{<:Dirichlet}, y::AbstractVector{<:Real})
    T = eltype(y)
    ϵ = _eps(T)

    res = forward(inv(td.transform), y)
    return logpdf(td.dist, mappedarray(x->x+ϵ, res.rv)) + res.logabsdetjac
end

# TODO: should eventually drop using `logpdf_with_trans` and replace with
# res = forward(inv(td.transform), y)
# logpdf(td.dist, res.rv) .- res.logabsdetjac
function _logpdf(td::MatrixTransformed, y::AbstractMatrix{<:Real})
    return logpdf_with_trans(td.dist, inv(td.transform)(y), true)
end

# rand
rand(td::UnivariateTransformed) = td.transform(rand(td.dist))
rand(rng::AbstractRNG, td::UnivariateTransformed) = td.transform(rand(rng, td.dist))

# These ovarloadings are useful for differentiating sampling wrt. params of `td.dist`
# or params of `Bijector`, as they are not inplace like the default `rand`
rand(td::MvTransformed) = td.transform(rand(td.dist))
rand(rng::AbstractRNG, td::MvTransformed) = td.transform(rand(rng, td.dist))
# TODO: implement more efficiently for flows
function rand(rng::AbstractRNG, td::MvTransformed, num_samples::Int)
    samples = rand(rng, td.dist, num_samples)
    res = reduce(hcat, map(axes(samples, 2)) do i
        return td.transform(view(samples, :, i))
    end)
    return res
end

function _rand!(rng::AbstractRNG, td::MvTransformed, x::AbstractVector{<:Real})
    rand!(rng, td.dist, x)
    x .= td.transform(x)
end

function _rand!(rng::AbstractRNG, td::MatrixTransformed, x::DenseMatrix{<:Real})
    rand!(rng, td.dist, x)
    x .= td.transform(x)
end

#############################################################
# Additional useful functions for `TransformedDistribution` #
#############################################################
"""
    logpdf_with_jac(td::UnivariateTransformed, y::Real)
    logpdf_with_jac(td::MvTransformed, y::AbstractVector{<:Real})
    logpdf_with_jac(td::MatrixTransformed, y::AbstractMatrix{<:Real})

Makes use of the `forward` method to potentially re-use computation
and returns a tuple `(logpdf, logabsdetjac)`.
"""
function logpdf_with_jac(td::UnivariateTransformed, y::Real)
    res = forward(inv(td.transform), y)
    return (logpdf(td.dist, res.rv) + res.logabsdetjac, res.logabsdetjac)
end

# TODO: implement more efficiently for flows in the case of `Matrix`
function logpdf_with_jac(td::MvTransformed, y::AbstractVector{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf(td.dist, res.rv) + res.logabsdetjac, res.logabsdetjac)
end

function logpdf_with_jac(td::MvTransformed, y::AbstractMatrix{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf(td.dist, res.rv) + res.logabsdetjac, res.logabsdetjac)
end

function logpdf_with_jac(td::MvTransformed{<:Dirichlet}, y::AbstractVector{<:Real})
    T = eltype(y)
    ϵ = _eps(T)

    res = forward(inv(td.transform), y)
    lp = logpdf(td.dist, mappedarray(x->x+ϵ, res.rv)) + res.logabsdetjac
    return (lp, res.logabsdetjac)
end

# TODO: should eventually drop using `logpdf_with_trans`
function logpdf_with_jac(td::MatrixTransformed, y::AbstractMatrix{<:Real})
    res = forward(inv(td.transform), y)
    return (logpdf_with_trans(td.dist, res.rv, true), res.logabsdetjac)
end

"""
    logpdf_forward(td::Transformed, x)
    logpdf_forward(td::Transformed, x, logjac)

Computes the `logpdf` using the forward pass of the bijector rather than using
the inverse transform to compute the necessary `logabsdetjac`.

This is similar to `logpdf_with_trans`.
"""
# TODO: implement more efficiently for flows in the case of `Matrix`
logpdf_forward(td::Transformed, x, logjac) = logpdf(td.dist, x) - logjac
logpdf_forward(td::Transformed, x) = logpdf_forward(td, x, logabsdetjac(td.transform, x))

function logpdf_forward(td::MvTransformed{<:Dirichlet}, x, logjac)
    T = eltype(x)
    ϵ = _eps(T)

    return logpdf(td.dist, mappedarray(z->z+ϵ, x)) - logjac
end


# forward function
const GLOBAL_RNG = Distributions.GLOBAL_RNG

function _forward(d::UnivariateDistribution, x)
    y, logjac = forward(Identity{0}(), x)
    return (x = x, y = y, logabsdetjac = logjac, logpdf = logpdf.(d, x))
end

forward(rng::AbstractRNG, d::Distribution) = _forward(d, rand(rng, d))
function forward(rng::AbstractRNG, d::Distribution, num_samples::Int)
    return _forward(d, rand(rng, d, num_samples))
end
function _forward(d::Distribution, x)
    y, logjac = forward(Identity{length(size(d))}(), x)
    return (x = x, y = y, logabsdetjac = logjac, logpdf = logpdf(d, x))
end

function _forward(td::Transformed, x)
    y, logjac = forward(td.transform, x)
    return (
        x = x,
        y = y,
        logabsdetjac = logjac,
        logpdf = logpdf_forward(td, x, logjac)
    )
end
function forward(rng::AbstractRNG, td::Transformed)
    return _forward(td, rand(rng, td.dist))
end
function forward(rng::AbstractRNG, td::Transformed, num_samples::Int)
    return _forward(td, rand(rng, td.dist, num_samples))
end

"""
    forward(d::Distribution)
    forward(d::Distribution, num_samples::Int)

Returns a `NamedTuple` with fields `x`, `y`, `logabsdetjac` and `logpdf`.

In the case where `d isa TransformedDistribution`, this means
- `x = rand(d.dist)`
- `y = d.transform(x)`
- `logabsdetjac` is the logabsdetjac of the "forward" transform.
- `logpdf` is the logpdf of `y`, not `x`

In the case where `d isa Distribution`, this means
- `x = rand(d)`
- `y = x`
- `logabsdetjac = 0.0`
- `logpdf` is logpdf of `x`
"""
forward(d::Distribution) = forward(GLOBAL_RNG, d)
forward(d::Distribution, num_samples::Int) = forward(GLOBAL_RNG, d, num_samples)

# utility stuff
Distributions.params(td::Transformed) = Distributions.params(td.dist)
function Base.maximum(td::UnivariateTransformed)
    # ordering might have changed, i.e. ub has been mapped to lb
    min, max = td.transform.((Base.minimum(td.dist), Base.maximum(td.dist)))
    return max > min ? max : min
end
function Base.minimum(td::UnivariateTransformed)
    # ordering might have changed, i.e. ub has been mapped to lb
    min, max = td.transform.((Base.minimum(td.dist), Base.maximum(td.dist)))
    return max < min ? max : min
end

# logabsdetjac for distributions
logabsdetjacinv(d::UnivariateDistribution, x::T) where T <: Real = zero(T)
logabsdetjacinv(d::MultivariateDistribution, x::AbstractVector{T}) where {T<:Real} = zero(T)


"""
    logabsdetjacinv(td::UnivariateTransformed, y::Real)
    logabsdetjacinv(td::MultivariateTransformed, y::AbstractVector{<:Real})

Computes the `logabsdetjac` of the _inverse_ transformation, since `rand(td)` returns
the _transformed_ random variable.
"""
logabsdetjacinv(td::UnivariateTransformed, y::Real) = logabsdetjac(inv(td.transform), y)
function logabsdetjacinv(td::MvTransformed, y::AbstractVector{<:Real})
    return logabsdetjac(inv(td.transform), y)
end
