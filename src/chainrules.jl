# differentation rule for the iterative algorithm in the inverse of `PlanarLayer`
ChainRulesCore.@scalar_rule(
    find_alpha(wt_y::Real, wt_u_hat::Real, b::Real),
    @setup(
        x = inv(1 + wt_u_hat * sech(Ω + b)^2),
    ),
    (x, - tanh(Ω + b) * x, x - 1),
)

function ChainRulesCore.rrule(::typeof(combine), m::PartitionMask, x_1, x_2, x_3)
    proj_x_1 = ChainRulesCore.ProjectTo(x_1)
    proj_x_2 = ChainRulesCore.ProjectTo(x_2)
    proj_x_3 = ChainRulesCore.ProjectTo(x_3)

    function combine_pullback(ΔΩ)
        Δ = ChainRulesCore.unthunk(ΔΩ)
        dx_1, dx_2, dx_3 = partition(m, Δ)
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), proj_x_1(dx_1), proj_x_2(dx_2), proj_x_3(dx_3)
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
    @inbounds for i = 1:length(r)
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
        @inbounds for j = 1:n - 1
            Δ_new[j] = (Δ[j] / r[j]) - (Δ[j + 1] / r[j + 1])
        end
        @inbounds Δ_new[n] = Δ[n] / r[n]

        return ChainRulesCore.NoTangent(), project_x(Δ_new)
    end

    y = similar(x)
    @inbounds y[1] = x[1]
    @inbounds for i = 2:length(x)
        y[i] = log(r[i])
    end

    return y, _transform_inverse_ordered_adjoint
end

function ChainRulesCore.rrule(::typeof(_transform_inverse_ordered), x::AbstractMatrix)
    # ensures that we remain in the primal's subspace
    project_x = ChainRulesCore.ProjectTo(x)

    r = similar(x)
    @inbounds for j = 1:size(x, 2), i = 1:size(x, 1)
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

        @inbounds for j = 1:size(Δ_new, 2), i = 1:n - 1
            Δ_new[i, j] = (Δ[i, j] / r[i, j]) - (Δ[i + 1, j] / r[i + 1, j])
        end

        @inbounds for j = 1:size(Δ_new, 2)
            Δ_new[n, j] = Δ[n, j] / r[n, j]
        end

        return ChainRulesCore.NoTangent(), project_x(Δ_new)
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

function ChainRulesCore.rrule(::typeof(_link_chol_lkj), W::UpperTriangular)
    K = LinearAlgebra.checksquare(W)
    N = ((K-1)*K) ÷ 2 
    
    z = zeros(eltype(W), N)
    tmp_vec = similar(z)

    idx = 1
    @inbounds for j = 2:K
        z[idx] = atanh(W[1, j])
        tmp = sqrt(1 - W[1, j]^2)
        tmp_vec[idx] = tmp
        idx += 1
        for i in 2:(j-1)
            p = W[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            tmp_vec[idx] = tmp
            z[idx] = atanh(p)
            idx += 1
        end
    end

    function pullback_link_chol_lkj(Δz_thunked)
        Δz = ChainRulesCore.unthunk(Δz_thunked)
        
        ΔW = similar(W)

        @inbounds ΔW[1,1] = zero(eltype(Δz))

        @inbounds for j=2:K
            idx_up_to_prev_column = ((j-1)*(j-2) ÷ 2)
            ΔW[j, j] = 0
            Δtmp = zero(eltype(Δz))
            for i in (j-1):-1:2
                tmp = tmp_vec[idx_up_to_prev_column + i - 1]
                p = W[i, j] / tmp
                ftmp = sqrt(1 - p^2)
                d_ftmp_p = -p / ftmp
                d_p_tmp = -W[i,j] / tmp^2

                Δp = Δz[idx_up_to_prev_column + i] / (1-p^2) + Δtmp * tmp * d_ftmp_p
                ΔW[i, j] = Δp / tmp
                Δtmp = Δp * d_p_tmp + Δtmp * ftmp 
            end
            ΔW[1, j] = Δz[idx_up_to_prev_column + 1] / (1-W[1,j]^2) - Δtmp / sqrt(1 - W[1,j]^2) * W[1,j]
        end

        return ChainRulesCore.NoTangent(), ΔW
    end

    return z, pullback_link_chol_lkj
end

function ChainRulesCore.rrule(::typeof(_link_chol_lkj), W::LowerTriangular)
    K = LinearAlgebra.checksquare(W)
    N = ((K-1)*K) ÷ 2 

    z = zeros(eltype(W), N)
    tmp_vec = similar(z)

    idx = 1
    @inbounds for i = 2:K
        z[idx] = atanh(W[i, 1])
        tmp = sqrt(1 - W[i, 1]^2)
        tmp_vec[idx] = tmp
        idx += 1
        for j in 2:(i-1)
            p = W[i, j] / tmp
            tmp *= sqrt(1 - p^2)
            tmp_vec[idx] = tmp
            z[idx] = atanh(p)
            idx += 1
        end
    end

    function pullback_link_chol_lkj(Δz_thunked)
        Δz = ChainRulesCore.unthunk(Δz_thunked)
        
        ΔW = similar(W)

        @inbounds ΔW[1,1] = zero(eltype(Δz))

        @inbounds for i=2:K
            idx_up_to_prev_row = ((i-1)*(i-2) ÷ 2)
            ΔW[i, i] = 0
            Δtmp = zero(eltype(Δz))
            for j in (i-1):-1:2
                tmp = tmp_vec[idx_up_to_prev_row + j - 1]
                p = W[i, j] / tmp
                ftmp = sqrt(1 - p^2)
                d_ftmp_p = -p / ftmp
                d_p_tmp = -W[i,j] / tmp^2

                Δp = Δz[idx_up_to_prev_row + j] / (1-p^2) + Δtmp * tmp * d_ftmp_p
                ΔW[i, j] = Δp / tmp
                Δtmp = Δp * d_p_tmp + Δtmp * ftmp 
            end
            ΔW[i, 1] = Δz[idx_up_to_prev_row + 1] / (1-W[i,1]^2) - Δtmp / sqrt(1 - W[i,1]^2) * W[i,1]
        end

        return ChainRulesCore.NoTangent(), ΔW
    end

    return z, pullback_link_chol_lkj
end

function ChainRulesCore.rrule(::typeof(_inv_link_chol_lkj), y::AbstractVector)
    K = _triu1_dim_from_length(length(y))
    
    W = similar(y, K, K)

    z_vec = similar(y)
    tmp_vec = similar(y)

    idx = 1
    @inbounds for j in 1:K
        W[1, j] = 1
        for i in 2:j
            z = tanh(y[idx])
            tmp = W[i-1, j]

            z_vec[idx] = z
            tmp_vec[idx] = tmp
            idx += 1

            W[i-1, j] = z * tmp
            W[i, j] = tmp * sqrt(1 - z^2)
        end
        for i in (j+1):K
            W[i, j] = 0
        end
    end

    function pullback_inv_link_chol_lkj(ΔW_thunked)
        ΔW = ChainRulesCore.unthunk(ΔW_thunked)
        
        Δy = zero(y)

        @inbounds for j in 1:K
            idx_up_to_prev_column = ((j-1)*(j-2) ÷ 2)
            Δtmp = ΔW[j,j]
            for i in j:-1:2
                idx = idx_up_to_prev_column + i - 1
                tmp = tmp_vec[idx]
                z = z_vec[idx]

                Δz = ΔW[i-1, j] * tmp - Δtmp * tmp / sqrt(1 - z^2) * z
                Δy[idx] = Δz / cosh(y[idx])^2
                Δtmp = ΔW[i-1, j] * z + Δtmp * sqrt(1 - z^2)
            end
        end

        return ChainRulesCore.NoTangent(), Δy
    end

    return W, pullback_inv_link_chol_lkj
end

function ChainRulesCore.rrule(::typeof(pd_from_upper), X::AbstractMatrix)
    return UpperTriangular(X)' * UpperTriangular(X), Δ -> begin
        Xu = UpperTriangular(X)
        return ChainRulesCore.NoTangent(), UpperTriangular(Xu * Δ + Xu * Δ')
    end
end

# Fixes Zygote's issues with `@debug`
ChainRulesCore.@non_differentiable _debug(::Any)
