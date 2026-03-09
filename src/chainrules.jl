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
function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractVector, ::Type{Ascending})
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

        return ChainRulesCore.NoTangent(), project_y(Δ_new), ChainRulesCore.NoTangent()
    end

    return _transform_ordered(y, Ascending), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractMatrix, ::Type{Ascending})
    # ensures that we remain in the primal's subspace
    project_y = ChainRulesCore.ProjectTo(y)

    function _transform_ordered_adjoint(ΔΩ)
        Δ_new = similar(y)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        @assert size(Δ) == size(Δ_new)
        n = size(Δ, 1)

        s = vec(sum(Δ; dims=1))
        Δ_new[1, :] .= s
        @inbounds for i in 2:n
            # Equivalent to
            #
            #    Δ_new[i] = sum(Δ[i:end]) * yexp[i - 1]
            #
            s .-= Δ[i - 1, :]
            Δ_new[i, :] .= s .* exp.(y[i, :])
        end

        return ChainRulesCore.NoTangent(), project_y(Δ_new), ChainRulesCore.NoTangent()
    end

    return _transform_ordered(y, Ascending), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractVector, ::Type{Descending})
    # ensures that we remain in the primal's subspace
    project_y = ChainRulesCore.ProjectTo(y)

    function _transform_ordered_adjoint(ΔΩ)
        Δ_new = similar(y)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        n = length(Δ)
        @assert n == length(Δ_new)

        # s = zero(eltype(Δ))
        s = Δ[1]
        @inbounds for i in 1:(n - 1)
            # s += Δ[i]
            Δ_new[i] = s * exp(y[i])
            s += Δ[i + 1]
        end
        # Δ_new[n] = s + Δ[n]
        Δ_new[n] = s

        return ChainRulesCore.NoTangent(), project_y(Δ_new), ChainRulesCore.NoTangent()
    end

    return _transform_ordered(y, Descending), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractMatrix, ::Type{Descending})
    # ensures that we remain in the primal's subspace
    project_y = ChainRulesCore.ProjectTo(y)

    function _transform_ordered_adjoint(ΔΩ)
        Δ_new = similar(y)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        n = size(Δ, 1)
        @assert size(Δ) == size(Δ_new)

        # s = zeros(eltype(Δ, size(Δ, 2)))
        s = Δ[1, :]
        @inbounds for i in 1:(n - 1)
            # s .+= Δ[i, :]
            Δ_new[i, :] .= s .* exp.(y[i, :])
            s .+= Δ[i + 1, :]
        end
        # Δ_new[n, :] = s .+ Δ[n, :]
        Δ_new[n, :] .= s

        return ChainRulesCore.NoTangent(), project_y(Δ_new), ChainRulesCore.NoTangent()
    end

    return _transform_ordered(y, Descending), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractVector, ::Type{FixedOrder{o}}) where {o}
    # ensures that we remain in the primal's subspace
    project_y = ChainRulesCore.ProjectTo(y)

    function _transform_ordered_adjoint(ΔΩ)
        Δ_new = similar(y)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        @assert length(Δ) == length(Δ_new)
        m = length(o)
        # Identity
        Δ_new .= Δ

        # Overwrite rows part of
        s = sum(Δ[collect(o)])
        Δ_new[o[1]] = s
        @inbounds for i in 2:m
            s -= Δ[o[i - 1]]
            Δ_new[o[i]] = s * exp(y[o[i]])
        end

        return ChainRulesCore.NoTangent(), project_y(Δ_new), ChainRulesCore.NoTangent()
    end

    return _transform_ordered(y, FixedOrder{o}), _transform_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_ordered), y::AbstractMatrix, ::Type{FixedOrder{o}}) where {o}
    # ensures that we remain in the primal's subspace
    project_y = ChainRulesCore.ProjectTo(y)

    function _transform_ordered_adjoint(ΔΩ)
        Δ_new = similar(y)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        @assert size(Δ) == size(Δ_new)
        
        Δ_new .= Δ
        
        m = length(o)
        
        # s = vec(sum(Δ[collect(o), :]; dims=1))
        # Δ_new[o[1], :] .= s        
        # @inbounds for i in 2:m
        #     s .-= Δ[o[i - 1], :]
        #     Δ_new[o[i], :] .= s .* exp.(y[o[i], :])
        # end

        s = Δ[o[m], :]
        @inbounds for i in m:-1:2
            Δ_new[o[i], :] .= s .* exp.(y[o[i], :])
            s .+= Δ[o[i - 1], :]
        end
        Δ_new[o[1], :] .= s

        return ChainRulesCore.NoTangent(), project_y(Δ_new), ChainRulesCore.NoTangent()
    end

    return _transform_ordered(y, FixedOrder{o}), _transform_ordered_adjoint
end


function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractVector, ::Type{Ascending})
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

        return ChainRulesCore.NoTangent(), project_x(Δ_new), ChainRulesCore.NoTangent()
    end

    y = similar(x)
    @inbounds y[1] = x[1]
    @inbounds for i in 2:length(x)
        y[i] = log(r[i])
    end

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractMatrix, ::Type{Ascending})
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

        return ChainRulesCore.NoTangent(), project_x(Δ_new), ChainRulesCore.NoTangent()
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

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractVector, ::Type{Descending})
    # ensures that we remain in the primal's subspace
    project_x = ChainRulesCore.ProjectTo(x)

    r = similar(x)
    n = length(r)
    @inbounds for i in 1:n
        if i == n
            r[i] = 1
        else
            r[i] = x[i] - x[i + 1]
        end
    end

    function _transform_inverse_ordered_adjoint(ΔΩ)
        Δ_new = similar(x)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        @assert length(Δ_new) == length(Δ)

        n = length(Δ_new)
        Δ_new[1] = Δ[1] / r[1]
        @inbounds for j in 2:n
            Δ_new[j] = (Δ[j] / r[j]) - (Δ[j-1] / r[j-1])
        end

        return ChainRulesCore.NoTangent(), project_x(Δ_new), ChainRulesCore.NoTangent()
    end

    y = similar(x)
    n = length(x)
    @inbounds for i in 1:(n - 1)
        y[i] = log(r[i])
    end
    @inbounds y[n] = x[n]

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractMatrix, ::Type{Descending})
    # ensures that we remain in the primal's subspace
    project_x = ChainRulesCore.ProjectTo(x)

    r = similar(x)
    n = size(x, 1)
    @inbounds for j in 1:size(x, 2), i in 1:n
        if i == n
            r[i, j] = 1
        else
            r[i, j] = x[i, j] - x[i + 1, j]
        end
    end

    function _transform_inverse_ordered_adjoint(ΔΩ)
        Δ_new = similar(x)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        n = size(Δ, 1)
        @assert size(Δ) == size(Δ_new)

        @inbounds for j in 1:size(Δ_new, 2)
            Δ_new[1, j] = Δ[1, j] / r[1, j]
        end

        @inbounds for j in 1:size(Δ_new, 2), i in 2:n
            Δ_new[i, j] = (Δ[i, j] / r[i, j]) - (Δ[i - 1, j] / r[i - 1, j])
        end

        return ChainRulesCore.NoTangent(), project_x(Δ_new), ChainRulesCore.NoTangent()
    end

    # Compute primal here so we can make use of the already
    # computed `r`.
    y = similar(x)
    n = size(x, 1)
    @inbounds for j in 1:size(x, 2), i in 1:n
        if i == n
            y[i, j] = x[i, j]
        else
            y[i, j] = log(r[i, j])
        end
    end

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractVector, ::Type{FixedOrder{o}}) where {o}
    # ensures that we remain in the primal's subspace
    project_x = ChainRulesCore.ProjectTo(x)

    m = length(o)
    r = zeros(eltype(x), m)
    @inbounds for i in 1:m
        if i == 1
            r[i] = 1
        else
            r[i] = x[o[i]] - x[o[i - 1]]
        end
    end

    function _transform_inverse_ordered_adjoint(ΔΩ)
        Δ_new = similar(x)
        Δ = ChainRulesCore.unthunk(ΔΩ)

        @assert length(Δ_new) == length(Δ)
        
        Δ_new .= Δ
        
        @inbounds for j in 1:(m - 1)
            Δ_new[o[j]] = (Δ[o[j]] / r[j]) - (Δ[o[j + 1]] / r[j + 1])
        end
        Δ_new[o[m]] = Δ[o[m]] / r[m]

        return ChainRulesCore.NoTangent(), project_x(Δ_new), ChainRulesCore.NoTangent()
    end

    y = copy(x)
    @inbounds for i in 2:m
        y[o[i]] = log(r[i])
    end

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractMatrix, ::Type{FixedOrder{o}}) where {o}
    # ensures that we remain in the primal's subspace
    project_x = ChainRulesCore.ProjectTo(x)

    m = length(o)
    
    r = zeros(eltype(x), m, size(x, 2))

    @inbounds for j in 1:size(x, 2), i in 1:m
        if i == 1
            r[i, j] = 1
        else
            r[i, j] = x[o[i], j] - x[o[i - 1], j]
        end
    end

    function _transform_inverse_ordered_adjoint(ΔΩ)
        Δ_new = similar(x)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        @assert size(Δ) == size(Δ_new)

        Δ_new .= Δ

        @inbounds for j in 1:size(Δ_new, 2), i in 1:(m - 1)
            Δ_new[o[i], j] = (Δ[o[i], j] / r[i, j]) - (Δ[o[i + 1], j] / r[i + 1, j])
        end

        @inbounds for j in 1:size(Δ_new, 2)
            Δ_new[o[m], j] = Δ[o[m], j] / r[m, j]
        end

        return ChainRulesCore.NoTangent(), project_x(Δ_new), ChainRulesCore.NoTangent()
    end

    # Compute primal here so we can make use of the already
    # computed `r`.
    y = copy(x)
    @inbounds for j in 1:size(x, 2), i in 2:m
        y[o[i], j] = log(r[i, j])
    end

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_link_chol_lkj_from_upper), W::AbstractMatrix)
    K = LinearAlgebra.checksquare(W)
    N = ((K - 1) * K) ÷ 2

    z = zeros(eltype(W), N)
    remainders = similar(z)

    starting_idx = 1
    @inbounds for j in 2:K
        z[starting_idx] = atanh(W[1, j])
        remainder_sq = W[j, j]^2
        starting_idx += 1
        for i in (j - 1):-1:2
            idx = starting_idx + i - 2
            remainder = sqrt(remainder_sq)
            remainders[idx] = remainder
            zt = W[i, j] / remainder
            z[idx] = asinh(zt)
            remainder_sq += W[i, j]^2
        end
        remainders[starting_idx - 1] = sqrt(remainder_sq)
        starting_idx += length((j - 1):-1:2)
    end

    function pullback_link_chol_lkj_from_upper(Δz_thunked)
        Δz = ChainRulesCore.unthunk(Δz_thunked)

        ΔW = similar(W)

        @inbounds ΔW[1, 1] = zero(eltype(Δz))

        @inbounds for j in 2:K
            idx_up_to_prev_column = ((j - 1) * (j - 2) ÷ 2)
            ΔW[j, j] = 0
            Δtmp = zero(eltype(Δz))
            for i in (j - 1):-1:2
                tmp = remainders[idx_up_to_prev_column + i - 1]
                p = W[i, j] / tmp
                ftmp = sqrt(1 - p^2)
                d_ftmp_p = -p / ftmp
                d_p_tmp = -W[i, j] / tmp^2

                Δp = Δz[idx_up_to_prev_column + i] / (1 - p^2) + Δtmp * tmp * d_ftmp_p
                ΔW[i, j] = Δp / tmp
                Δtmp = Δp * d_p_tmp + Δtmp * ftmp
            end
            ΔW[1, j] =
                Δz[idx_up_to_prev_column + 1] / (1 - W[1, j]^2) -
                Δtmp / sqrt(1 - W[1, j]^2) * W[1, j]
        end

        return ChainRulesCore.NoTangent(), ΔW
    end

    return z, pullback_link_chol_lkj_from_upper
end

function ChainRulesCore.rrule(::typeof(_link_chol_lkj_from_lower), W::AbstractMatrix)
    K = LinearAlgebra.checksquare(W)
    N = ((K - 1) * K) ÷ 2

    z = zeros(eltype(W), N)
    remainders = similar(z)

    starting_idx = 1
    @inbounds for i in 2:K
        z[starting_idx] = atanh(W[i, 1])
        remainder_sq = W[i, i]^2
        starting_idx += 1
        for j in (i - 1):-1:2
            idx = starting_idx + j - 2
            remainder = sqrt(remainder_sq)
            remainders[idx] = remainder
            zt = W[i, j] / remainder
            z[idx] = asinh(zt)
            remainder_sq += W[i, j]^2
        end
        remainders[starting_idx - 1] = sqrt(remainder_sq)
        starting_idx += length((i - 1):-1:2)
    end

    function pullback_link_chol_lkj_from_lower(Δz_thunked)
        Δz = ChainRulesCore.unthunk(Δz_thunked)

        ΔW = similar(W)

        @inbounds ΔW[1, 1] = zero(eltype(Δz))

        @inbounds for i in 2:K
            idx_up_to_prev_row = ((i - 1) * (i - 2) ÷ 2)
            ΔW[i, i] = 0
            Δtmp = zero(eltype(Δz))
            for j in (i - 1):-1:2
                tmp = remainders[idx_up_to_prev_row + j - 1]
                p = W[i, j] / tmp
                ftmp = sqrt(1 - p^2)
                d_ftmp_p = -p / ftmp
                d_p_tmp = -W[i, j] / tmp^2

                Δp = Δz[idx_up_to_prev_row + j] / (1 - p^2) + Δtmp * tmp * d_ftmp_p
                ΔW[i, j] = Δp / tmp
                Δtmp = Δp * d_p_tmp + Δtmp * ftmp
            end
            ΔW[i, 1] =
                Δz[idx_up_to_prev_row + 1] / (1 - W[i, 1]^2) -
                Δtmp / sqrt(1 - W[i, 1]^2) * W[i, 1]
        end

        return ChainRulesCore.NoTangent(), ΔW
    end

    return z, pullback_link_chol_lkj_from_lower
end

function ChainRulesCore.rrule(::typeof(_inv_link_chol_lkj), y::AbstractVector)
    W_logJ, back = _inv_link_chol_lkj_rrule(y)

    function pullback_inv_link_chol_lkj(ΔW_ΔlogJ)
        Δy = back(ChainRulesCore.unthunk(ΔW_ΔlogJ))
        return ChainRulesCore.NoTangent(), Δy
    end

    return W_logJ, pullback_inv_link_chol_lkj
end

function ChainRulesCore.rrule(::typeof(pd_from_upper), X::AbstractMatrix)
    return UpperTriangular(X)' * UpperTriangular(X),
    Δ_thunked -> begin
        Δ = ChainRulesCore.unthunk(Δ_thunked)
        Xu = UpperTriangular(X)
        return ChainRulesCore.NoTangent(), UpperTriangular(Xu * Δ + Xu * Δ')
    end
end

# Fixes AD issues with `@debug`
ChainRulesCore.@non_differentiable _debug(::Any)
