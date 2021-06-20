# differentation rule for the iterative algorithm in the inverse of `PlanarLayer`
ChainRulesCore.@scalar_rule(
    find_alpha(wt_y::Real, wt_u_hat::Real, b::Real),
    @setup(
        x = inv(1 + wt_u_hat * sech(Ω + b)^2),
    ),
    (x, - tanh(Ω + b) * x, x - 1),
)

# `OrderedBijector`
function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractVector)
    function _transform_ordered_adjoint(Δ)
        Δ_new = similar(y)
        n = length(Δ)
        @assert n == length(Δ_new)

        s = sum(Δ)
        Δ_new[1] = s
        @inbounds for i in 2:n
            # Equivalent to
            #
            #    Δ_new[i] = sum(Δ[i:end]) * yexp[i - 1]
            #
            s -= Δ[i - 1]
            Δ_new[i] = s * exp(y[i])
        end

        # Using `NO_FIELDS` to be backwards-compatible.
        return (ChainRulesCore.NO_FIELDS, Δ_new)
    end

    return _transform_ordered(y), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractMatrix)
    function _transform_ordered_adjoint(Δ)
        Δ_new = similar(y)
        n = length(Δ)
        @assert n == size(Δ_new, 1)

        s = sum(Δ)
        Δ_new[1, :] .= s
        @inbounds for i in 2:n
            # Equivalent to
            #
            #    Δ_new[i] = sum(Δ[i:end]) * yexp[i - 1]
            #
            s -= Δ[i - 1]
            Δ_new[i, :] = s * exp.(y[i, :])
        end

        return (ChainRulesCore.NO_FIELDS, Δ_new)
    end

    return _transform_ordered(y), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractVector)
    r = similar(x)
    @inbounds for i = 1:length(r)
        if i == 1
            r[i] = 1
        else
            r[i] = x[i] - x[i - 1]
        end
    end

    function _transform_inverse_ordered_adjoint(Δ)
        Δ_new = similar(x)
        @assert length(Δ_new) == length(Δ)

        n = length(Δ_new)
        @inbounds for j = 1:n - 1
            Δ_new[j] = (Δ[j] / r[j]) - (Δ[j + 1] / r[j + 1])
        end
        @inbounds Δ_new[n] = Δ[n] / r[n]

        return (ChainRulesCore.NO_FIELDS, Δ_new)
    end

    y = similar(x)
    @inbounds y[1] = x[1]
    @inbounds for i = 2:length(x)
        y[i] = log(r[i])
    end

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractMatrix)
    r = similar(x)
    @inbounds for j = 1:size(x, 2), i = 1:size(x, 1)
        if i == 1
            r[i, j] = 1
        else
            r[i, j] = x[i, j] - x[i - 1, j]
        end
    end

    function _transform_inverse_ordered_adjoint(Δ)
        Δ_new = similar(x)
        n = length(Δ)
        @assert n == size(Δ_new, 1)

        @inbounds for j = 1:size(Δ_new, 2), i = 1:n - 1
            Δ_new[i, j] = (Δ[i, j] / r[i, j]) - (Δ[i + 1, j] / r[i + 1, j])
        end

        @inbounds for j = 1:size(Δ_new)
            Δ_new[n, j] = Δ[n, j] / r[n, j]
        end

        return (ChainRulesCore.NO_FIELDS, Δ_new)
    end

    # Compute primal here so we can make use of the already
    # computed `r`.
    y = similar(x)
    @inbounds for j = 1:size(x, 2), i = 1:size(x, 1)
        if i == 1
            y[i, j] = x[i, j]
        else
            y[i, j] = log(r[i, j])
        end
    end

    return y, _transform_inverse_ordered_adjoint
end
