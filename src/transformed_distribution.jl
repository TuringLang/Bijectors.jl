function variateform(d::Distribution, b)
    sz_out = output_size(b, d)
    return ArrayLikeVariate{length(sz_out)}
end

variateform(::MultivariateDistribution, ::Inverse{VecCholeskyBijector}) = CholeskyVariate

# Transformed distributions
struct TransformedDistribution{D,B,V} <:
       Distribution{V,Continuous} where {D<:ContinuousDistribution,B}
    dist::D
    transform::B

    function TransformedDistribution(d::ContinuousDistribution, b)
        return new{typeof(d),typeof(b),variateform(d, b)}(d, b)
    end
end

# fields may contain nested numerical parameters
Functors.@functor TransformedDistribution

const UnivariateTransformed = TransformedDistribution{<:Distribution,<:Any,Univariate}
const MultivariateTransformed = TransformedDistribution{<:Distribution,<:Any,Multivariate}
const MvTransformed = MultivariateTransformed
const MatrixTransformed = TransformedDistribution{<:Distribution,<:Any,Matrixvariate}
const Transformed = TransformedDistribution

"""
    transformed(d::Distribution)
    transformed(d::Distribution, b::Bijector)

Couples distribution `d` with the bijector `b` by returning a `TransformedDistribution`.

If no bijector is provided, i.e. `transformed(d)` is called, then 
`transformed(d, bijector(d))` is returned.
"""
transformed(d::Distribution, b) = TransformedDistribution(d, b)
transformed(d) = transformed(d, bijector(d))

"""
    bijector(d::Distribution)

Returns the constrained-to-unconstrained bijector for distribution `d`.
"""
function bijector(td::TransformedDistribution)
    b = bijector(td.dist)
    return b === identity ? inverse(td.transform) : b ∘ inverse(td.transform)
end

"""
    has_constant_bijector(dist_type::Type)

Returns `true` if the distribution type `dist_type` has a constant bijector,
i.e. the return-value of [`bijector`](@ref) does not depend on runtime information.
"""
has_constant_bijector(d::Type) = false
has_constant_bijector(d::Type{<:Normal}) = true
has_constant_bijector(d::Type{<:Distributions.AbstractMvNormal}) = true
has_constant_bijector(d::Type{<:Distributions.AbstractMvLogNormal}) = true
has_constant_bijector(d::Type{<:TDist}) = true
has_constant_bijector(d::Type{<:Distributions.GenericMvTDist}) = true
has_constant_bijector(d::Type{<:PositiveDistribution}) = true
has_constant_bijector(d::Type{<:SimplexDistribution}) = true
has_constant_bijector(d::Type{<:KSOneSided}) = true
function has_constant_bijector(::Type{<:Product{Continuous,D}}) where {D}
    return has_constant_bijector(D)
end
function has_constant_bijector(
    ::Type{<:Distributions.ProductDistribution{<:Any,<:Any,A}}
) where {A}
    return has_constant_bijector(eltype(A))
end

# Container distributions.
bijector(d::DiscreteUnivariateDistribution) = identity
bijector(d::DiscreteMultivariateDistribution) = identity
bijector(d::ContinuousUnivariateDistribution) = TruncatedBijector(minimum(d), maximum(d))
bijector(d::Product{Discrete}) = identity
function bijector(d::Product{Continuous})
    D = eltype(d.v)
    return if has_constant_bijector(D)
        elementwise(bijector(d.v[1]))
    else
        # FIXME: This is not great. Should use something like
        # `Stacked(map(bijector, d.v))` instead.
        # TODO: Specialize. F.ex. for FillArrays.jl we can do much better.
        TruncatedBijector(_minmax(d.v)...)
    end
end

@generated function _minmax(d::AbstractArray{T}) where {T}
    try
        min, max = minimum(T), maximum(T)
        return :($min, $max)
    catch
        return :(minimum.(d), maximum.(d))
    end
end

function bijector(d::Distributions.ProductDistribution{N,0,A}) where {N,A}
    # This is the univariate scenario, so if we have a constant bijector
    # we can just use the same one for all elements.
    return if has_constant_bijector(eltype(A))
        elementwise(bijector(d.dists[1]))
    else
        ProductBijector(map(bijector, d.dists))
    end
end

function bijector(d::Distributions.ProductDistribution{N,M,A}) where {N,M,A}
    dists = d.dists
    bs = bijector.(dists)
    return ProductBijector{typeof(bs),N - M}(bs)
end

# Specialized implementations.
bijector(d::Normal) = identity
bijector(d::Distributions.AbstractMvNormal) = identity
bijector(d::Distributions.AbstractMvLogNormal) = elementwise(log)
bijector(d::TDist) = identity
bijector(d::Distributions.GenericMvTDist) = identity
bijector(d::PositiveDistribution) = elementwise(log)
bijector(d::SimplexDistribution) = SimplexBijector()
bijector(d::KSOneSided) = Logit(zero(eltype(d)), one(eltype(d)))

bijector_bounded(d, a=minimum(d), b=maximum(d)) = Logit(a, b)
bijector_lowerbounded(d, a=minimum(d)) = elementwise(log) ∘ Shift(-a)
function bijector_upperbounded(d, b=maximum(d))
    return elementwise(log) ∘ Shift(b) ∘ Scale(-one(typeof(b)))
end

const BoundedDistribution = Union{Arcsine,Biweight,Cosine,Epanechnikov,Beta,NoncentralBeta}
bijector(d::BoundedDistribution) = bijector_bounded(d)

const LowerboundedDistribution = Union{Pareto,Levy}
bijector(d::LowerboundedDistribution) = bijector_lowerbounded(d)

bijector(d::PDMatDistribution) = PDVecBijector()
bijector(d::MatrixBeta) = PDVecBijector()

bijector(d::LKJ) = VecCorrBijector()
bijector(d::LKJCholesky) = VecCholeskyBijector(d.uplo)

function bijector(d::Distributions.ReshapedDistribution)
    inner_dims = size(d.dist)
    outer_dims = d.dims
    b = Reshape(outer_dims, inner_dims)
    return inverse(b) ∘ bijector(d.dist) ∘ b
end

##############################
# Distributions.jl interface #
##############################

# size
Base.length(td::Transformed) = prod(output_size(td.transform, size(td.dist)))
Base.size(td::Transformed) = output_size(td.transform, size(td.dist))

function logpdf(td::UnivariateTransformed, y::Real)
    x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
    return logpdf(td.dist, x) + logjac
end

# TODO: implement more efficiently for flows in the case of `Matrix`
function logpdf(td::MvTransformed, y::AbstractMatrix{<:Real})
    # batch-implementation for multivariate
    x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
    return logpdf(td.dist, x) + logjac
end

function logpdf(td::MvTransformed{<:Dirichlet}, y::AbstractMatrix{<:Real})
    T = eltype(y)
    ϵ = _eps(T)

    x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
    return logpdf(td.dist, mappedarray(x -> x + ϵ, x)) + logjac
end

function logpdf(
    td::TransformedDistribution{T}, y::AbstractVector{<:Real}
) where {T<:Union{LKJ,LKJCholesky}}
    x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
    return logpdf(td.dist, x) + logjac
end

function _logpdf(td::MvTransformed, y::AbstractVector{<:Real})
    x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
    return logpdf(td.dist, x) + logjac
end

function _logpdf(td::MvTransformed{<:Dirichlet}, y::AbstractVector{<:Real})
    T = eltype(y)
    ϵ = _eps(T)

    x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
    return logpdf(td.dist, mappedarray(x -> x + ϵ, x)) + logjac
end

# TODO: should eventually drop using `logpdf_with_trans` and replace with
# x, logjac = with_logabsdet_jacobian(inverse(td.transform), y)
# logpdf(td.dist, x) .- logjac
function _logpdf(td::MatrixTransformed, y::AbstractMatrix{<:Real})
    return logpdf_with_trans(td.dist, inverse(td.transform)(y), true)
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
    res = reduce(
        hcat,
        map(axes(samples, 2)) do i
            return td.transform(view(samples, :, i))
        end,
    )
    return res
end

function _rand!(rng::AbstractRNG, td::MvTransformed, x::AbstractVector{<:Real})
    rand!(rng, td.dist, x)
    return x .= td.transform(x)
end

function _rand!(rng::AbstractRNG, td::MatrixTransformed, x::DenseMatrix{<:Real})
    rand!(rng, td.dist, x)
    return x .= td.transform(x)
end

function rand(
    rng::AbstractRNG, td::TransformedDistribution{T}
) where {T<:Union{LKJ,LKJCholesky}}
    return td.transform(rand(rng, td.dist))
end

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
