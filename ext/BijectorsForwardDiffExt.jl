module BijectorsForwardDiffExt

import Bijectors: Bijectors, find_alpha, _value_and_gradient, _value_and_jacobian
import ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

function _value_and_gradient(
    f, backend::AutoForwardDiff{chunksize,T}, x::AbstractVector
) where {chunksize,T}
    if isempty(x)
        return f(x), similar(x, 0)
    end
    result = ForwardDiff.DiffResults.GradientResult(x)
    chunk = isnothing(chunksize) ? ForwardDiff.Chunk(x) : ForwardDiff.Chunk{chunksize}()
    tag = T === Nothing ? ForwardDiff.Tag(f, eltype(x)) : backend.tag
    config = ForwardDiff.GradientConfig(nothing, x, chunk, tag)
    if T === Nothing
        ForwardDiff.checktag(config, f, x)
    end
    ForwardDiff.gradient!(result, f, x, config, Val(false))
    return ForwardDiff.DiffResults.value(result), ForwardDiff.DiffResults.gradient(result)
end

function _value_and_jacobian(
    f, backend::AutoForwardDiff{chunksize,T}, x::AbstractVector
) where {chunksize,T}
    y = f(x)
    if isempty(x)
        return y, Matrix{eltype(y)}(undef, length(y), 0)
    end
    result = ForwardDiff.DiffResults.JacobianResult(y, x)
    chunk = isnothing(chunksize) ? ForwardDiff.Chunk(x) : ForwardDiff.Chunk{chunksize}()
    tag = T === Nothing ? ForwardDiff.Tag(f, eltype(x)) : backend.tag
    config = ForwardDiff.JacobianConfig(nothing, x, chunk, tag)
    if T === Nothing
        ForwardDiff.checktag(config, f, x)
    end
    ForwardDiff.jacobian!(result, f, x, config, Val(false))
    return ForwardDiff.DiffResults.value(result), ForwardDiff.DiffResults.jacobian(result)
end

Bijectors._eps(::Type{<:ForwardDiff.Dual{<:Any,Real}}) = Bijectors._eps(Real)
Bijectors._eps(::Type{<:ForwardDiff.Dual{<:Any,<:Integer}}) = Bijectors._eps(Real)

# Define forward-mode rule for ForwardDiff and don't trust support for ForwardDiff in Roots
# https://github.com/JuliaMath/Roots.jl/issues/314
function Bijectors.find_alpha(
    wt_y::ForwardDiff.Dual{T,<:Real},
    wt_u_hat::ForwardDiff.Dual{T,<:Real},
    b::ForwardDiff.Dual{T,<:Real},
) where {T}
    # Compute primal
    value_wt_y = ForwardDiff.value(wt_y)
    value_wt_u_hat = ForwardDiff.value(wt_u_hat)
    value_b = ForwardDiff.value(b)
    Ω = find_alpha(value_wt_y, value_wt_u_hat, value_b)

    # Compute derivative
    partials_wt_y = ForwardDiff.partials(wt_y)
    partials_wt_u_hat = ForwardDiff.partials(wt_u_hat)
    partials_b = ForwardDiff.partials(b)
    x = inv(1 + value_wt_u_hat * sech(Ω + value_b)^2)
    ∂Ω = x * (partials_wt_y - tanh(Ω + value_b) * partials_wt_u_hat) + (x - 1) * partials_b

    return ForwardDiff.Dual{T}(Ω, ∂Ω)
end

end
