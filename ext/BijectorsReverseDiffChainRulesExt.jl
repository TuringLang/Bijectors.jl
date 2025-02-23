module BijectorsReverseDiffChainRulesExt

using ReverseDiff: @grad, value, track, TrackedMatrix

using ChainRules: ChainRules

import Bijectors: lower_triangular, upper_triangular, cholesky_lower, cholesky_upper

using Bijectors.LinearAlgebra

@grad function cholesky_lower(X_tracked::TrackedMatrix)
    X = value(X_tracked)
    H, hermitian_pullback = ChainRules.rrule(Hermitian, X, :L)
    C, cholesky_pullback = ChainRules.rrule(cholesky, H, Val(false))
    function cholesky_lower_pullback(ΔL)
        ΔC = ChainRules.Tangent{typeof(C)}(; factors=(C.uplo === :L ? ΔL : ΔL'))
        ΔH = cholesky_pullback(ΔC)[2]
        Δx = hermitian_pullback(ΔH)[2]
        # No need to add pullback for `lower_triangular`, because the pullback
        # for `Hermitian` already produces the correct result (i.e. the lower-triangular
        # part zeroed out).
        return (Δx,)
    end

    return lower_triangular(parent(C.L)), cholesky_lower_pullback
end

@grad function cholesky_upper(X_tracked::TrackedMatrix)
    X = value(X_tracked)
    H, hermitian_pullback = ChainRules.rrule(Hermitian, X, :U)
    C, cholesky_pullback = ChainRules.rrule(cholesky, H, Val(false))
    function cholesky_upper_pullback(ΔU)
        ΔC = ChainRules.Tangent{typeof(C)}(; factors=(C.uplo === :U ? ΔU : ΔU'))
        ΔH = cholesky_pullback(ΔC)[2]
        Δx = hermitian_pullback(ΔH)[2]
        # No need to add pullback for `upper_triangular`, because the pullback
        # for `Hermitian` already produces the correct result (i.e. the upper-triangular
        # part zeroed out).
        return (Δx,)
    end

    return upper_triangular(parent(C.U)), cholesky_upper_pullback
end

end
