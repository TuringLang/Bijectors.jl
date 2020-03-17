const RTR = ReverseDiff.TrackedReal
const RTV = ReverseDiff.TrackedVector
const RTM = ReverseDiff.TrackedMatrix
using ReverseDiff: record_mul
using ReverseDiff: SpecialInstruction

_eps(::Type{<:RTR{T}}) where {T} = _eps(T)

Base.:*(A::RTM{<:Any, D}, B::AbstractTriangular) where {D} = record_mul(A, B, D)
Base.:*(A::AbstractTriangular, B::RTV{<:Any, D}) where {D} = record_mul(A, B, D)
Base.:*(A::AbstractTriangular, B::RTM{<:Any, D}) where {D} = record_mul(A, B, D)
function Base.:*(A::Adjoint{T, <:AbstractTriangular{T}}, B::RTM{<:Any, D}) where {T, D}
    return record_mul(A, B, D)
end
function Base.:*(A::Adjoint{T, <:AbstractTriangular{T}}, B::RTV{<:Any, D}) where {T, D}
    return record_mul(A, B, D)
end

function replace_diag(::typeof(log), X::RTM{<:Any, D}) where {D}
    tp = ReverseDiff.tape(X)
    X_value = ReverseDiff.value(X)
    f(i, j) = i == j ? log(X_value[i, j]) : X_value[i, j]
    out_value = f.(1:size(X_value, 1), (1:size(X_value, 2))')
    function back(∇)
        g(i, j) = i == j ? ∇[i, j]/X_value[i, j] : ∇[i, j]
        return g.(1:size(X_value, 1), (1:size(X_value, 2))')
    end
    out = ReverseDiff.track(out_value, D, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, replace_diag, (log, X), out, (back,))
    return out
end

function replace_diag(::typeof(exp), X::RTM{<:Any, D}) where {D}
    tp = ReverseDiff.tape(X)
    X_value = ReverseDiff.value(X)
    f(i, j) = i == j ? exp(X_value[i, j]) : X_value[i, j]
    out_value = f.(1:size(X_value, 1), (1:size(X_value, 2))')
    function back(∇)
        g(i, j) = i == j ? ∇[i, j]*exp(X_value[i, j]) : ∇[i, j]
        return g.(1:size(X_value, 1), (1:size(X_value, 2))')
    end
    out = ReverseDiff.track(out_value, D, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, replace_diag, (exp, X), out, (back,))
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(replace_diag)})
    output = instruction.output
    input = instruction.input
    input_deriv = ReverseDiff.deriv(input[2])
    P = instruction.cache[1]
    input_deriv .+= P(ReverseDiff.deriv(output))
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(replace_diag)})
    output, input = instruction.output, instruction.input
    out_value = replace_diag(ReverseDiff.value(input[1]), ReverseDiff.value(input[2]))
    ReverseDiff.value!(output, out_value)
    return nothing
end
