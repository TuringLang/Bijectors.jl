# non-bijector from an angle to a circle
# its "inverse" is a left inverse from the plane to an angle such that
# (inv(Angle()) ∘ Angle())(θ) == θ and (Angle() ∘ inv(Angle()))(z) == normalize(z)
# Instead of the logdetjac, we use that a standard bivariate normally distributed random
# variable `z` upon normalization is uniformly distributed on the circle.
# so the necessary adjustment to the logpdf by using `z` is -‖z‖^2/2
# see https://mc-stan.org/docs/2_26/reference-manual/unit-vector-section.html

struct Angle <: Bijector{0} end

(b::Angle)(θ::Real) = [sincos(θ)...]

(ib::Inverse{<:Angle})(z::AbstractVector{<:Real}) = atan(z...)

logabsdetjac(::Angle, θ::Real) = one(θ) / 2

function logabsdetjac(::Inverse{<:Angle}, z::AbstractVector{<:Real})
    y, x = z
    return -(x^2 + y^2) / 2
end
