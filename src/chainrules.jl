# differentation rule for the iterative algorithm in the inverse of `PlanarLayer`
ChainRulesCore.@scalar_rule(
    find_alpha(wt_y::Real, wt_u_hat::Real, b::Real),
    @setup(x = inv(1 + wt_u_hat * sech(Ω + b)^2),),
    (x, -tanh(Ω + b) * x, x - 1),
)

function ChainRulesCore.rrule(::typeof(combine), m::PartitionMask, x_1, x_2, x_3)
    proj_x_1 = ChainRulesCore.ProjectTo(x_1)
    proj_x_2 = ChainRulesCore.ProjectTo(x_2)
    proj_x_3 = ChainRulesCore.ProjectTo(x_3)

    function combine_pullback(ΔΩ)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        dx_1, dx_2, dx_3 = partition(m, Δ)
        return ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent(), proj_x_1(dx_1), proj_x_2(dx_2),
        proj_x_3(dx_3)
    end

    return combine(m, x_1, x_2, x_3), combine_pullback
end

# `OrderedBijector`
function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractVector)
    # ensures that we remain in the primal's subspace
    project_y = ChainRulesCore.ProjectTo(y)

    function _transform_ordered_adjoint(ΔΩ)
        Δ_new = similar(y)
        Δ = ChainRulesCore.unthunk(ΔΩ)
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

        return ChainRulesCore.NoTangent(), project_y(Δ_new)
    end

    return _transform_ordered(y), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractMatrix)
    # ensures that we remain in the primal's subspace
    project_y = ChainRulesCore.ProjectTo(y)

    function _transform_ordered_adjoint(ΔΩ)
        Δ_new = similar(y)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        n = size(Δ, 1)
        @assert size(Δ) == size(Δ_new)

        s = vec(sum(Δ; dims=1))
        Δ_new[1, :] .= s
        @inbounds for i in 2:n
            # Equivalent to
            #
            #    Δ_new[i] = sum(Δ[i:end]) * yexp[i - 1]
            #
            s -= Δ[i - 1, :]
            Δ_new[i, :] = s .* exp.(y[i, :])
        end

        return ChainRulesCore.NoTangent(), project_y(Δ_new)
    end

    return _transform_ordered(y), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractVector)
    # ensures that we remain in the primal's subspace
    project_x = ChainRulesCore.ProjectTo(x)

    r = similar(x)
    @inbounds for i in 1:length(r)
        if i == 1
            r[i] = 1
        else
            r[i] = x[i] - x[i - 1]
        end
    end

    function _transform_inverse_ordered_adjoint(ΔΩ)
        Δ_new = similar(x)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        @assert length(Δ_new) == length(Δ)

        n = length(Δ_new)
        @inbounds for j in 1:(n - 1)
            Δ_new[j] = (Δ[j] / r[j]) - (Δ[j + 1] / r[j + 1])
        end
        @inbounds Δ_new[n] = Δ[n] / r[n]

        return ChainRulesCore.NoTangent(), project_x(Δ_new)
    end

    y = similar(x)
    @inbounds y[1] = x[1]
    @inbounds for i in 2:length(x)
        y[i] = log(r[i])
    end

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractMatrix)
    # ensures that we remain in the primal's subspace
    project_x = ChainRulesCore.ProjectTo(x)

    r = similar(x)
    @inbounds for j in 1:size(x, 2), i in 1:size(x, 1)
        if i == 1
            r[i, j] = 1
        else
            r[i, j] = x[i, j] - x[i - 1, j]
        end
    end

    function _transform_inverse_ordered_adjoint(ΔΩ)
        Δ_new = similar(x)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        n = size(Δ, 1)
        @assert size(Δ) == size(Δ_new)

        @inbounds for j in 1:size(Δ_new, 2), i in 1:(n - 1)
            Δ_new[i, j] = (Δ[i, j] / r[i, j]) - (Δ[i + 1, j] / r[i + 1, j])
        end

        @inbounds for j in 1:size(Δ_new, 2)
            Δ_new[n, j] = Δ[n, j] / r[n, j]
        end

        return ChainRulesCore.NoTangent(), project_x(Δ_new)
    end

    # Compute primal here so we can make use of the already
    # computed `r`.
    y = similar(x)
    @inbounds for j in 1:size(x, 2), i in 1:size(x, 1)
        if i == 1
            y[i, j] = x[i, j]
        else
            y[i, j] = log(r[i, j])
        end
    end

    return y, _transform_inverse_ordered_adjoint
end

# Fixes Zygote's issues with `@debug`
ChainRulesCore.@non_differentiable _debug(::Any)
