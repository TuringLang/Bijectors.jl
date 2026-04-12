module BijectorsEnzymeExt

import Bijectors: _value_and_gradient, _value_and_jacobian
import ADTypes: AutoEnzyme
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore

const AnyFunctionDuplicated = Union{
    EnzymeCore.Duplicated,EnzymeCore.DuplicatedNoNeed,EnzymeCore.MixedDuplicated
}

# `AutoEnzyme()` leaves mode selection to the operation, matching the old DI-backed
# behaviour: reverse mode for scalar gradients, forward mode for Jacobians.
_gradient_mode(backend::AutoEnzyme{<:EnzymeCore.ForwardMode}) = backend.mode
_gradient_mode(backend::AutoEnzyme{<:EnzymeCore.ReverseMode}) = backend.mode
_gradient_mode(::AutoEnzyme{Nothing}) = Enzyme.Reverse

_forward_withprimal_mode(backend::AutoEnzyme) = Enzyme.WithPrimal(_gradient_mode(backend))

function _gradient_withprimal_mode(backend::AutoEnzyme{<:EnzymeCore.ReverseMode})
    return EnzymeCore.WithPrimal(backend.mode)
end
_gradient_withprimal_mode(::AutoEnzyme{Nothing}) = Enzyme.ReverseWithPrimal

_jacobian_mode(backend::AutoEnzyme) = backend.mode
_jacobian_mode(::AutoEnzyme{Nothing}) = Enzyme.Forward
_jacobian_withprimal_mode(backend::AutoEnzyme) = Enzyme.WithPrimal(_jacobian_mode(backend))

_enzyme_annotate_f(f, ::AutoEnzyme{M,Nothing}, mode) where {M} = f
_enzyme_annotate_f(f, ::AutoEnzyme{M,<:EnzymeCore.Const}, mode) where {M} = Enzyme.Const(f)
function _enzyme_duplicated_function(f, ::Type{<:EnzymeCore.MixedDuplicated})
    return Enzyme.Duplicated(f, Enzyme.make_zero(f))
end
function _enzyme_duplicated_function(f, ::Type{A}) where {A<:AnyFunctionDuplicated}
    return A(f, Enzyme.make_zero(f))
end

function _enzyme_annotate_f(f, ::AutoEnzyme{M,A}, mode) where {M,A<:AnyFunctionDuplicated}
    # Mutable functors need a duplicated function annotation plus a shadow.
    if Enzyme.guess_activity(typeof(f), mode) <: EnzymeCore.Const
        return Enzyme.Const(f)
    else
        return _enzyme_duplicated_function(f, A)
    end
end
function _enzyme_annotate_f(f, ::AutoEnzyme{M,A}, mode) where {M,A<:EnzymeCore.Annotation}
    throw(ArgumentError("unsupported Enzyme function annotation $A"))
end

# For reverse mode (explicit or auto/nothing), use ReverseWithPrimal so value and
# gradient are computed in a single autodiff call rather than evaluating f twice.
function _value_and_gradient(
    f,
    backend::Union{AutoEnzyme{Nothing},AutoEnzyme{<:EnzymeCore.ReverseMode}},
    x::AbstractVector,
)
    mode = _gradient_withprimal_mode(backend)
    annotated_f = _enzyme_annotate_f(f, backend, mode)
    dx = zero(x)
    _, val = Enzyme.autodiff(mode, annotated_f, Enzyme.Active, Enzyme.Duplicated(x, dx))
    return val, dx
end

# For forward mode the gradient already requires O(n) JVPs; one extra f(x) evaluation
# is negligible.
function _value_and_gradient(f, backend::AutoEnzyme, x::AbstractVector)
    mode = _forward_withprimal_mode(backend)
    grad = similar(x)
    fill!(grad, zero(eltype(x)))
    value = nothing
    for i in eachindex(x)
        dx = zero(x)
        dx[i] = one(eltype(x))
        directional, primal = Enzyme.autodiff(
            mode, _enzyme_annotate_f(f, backend, mode), Enzyme.Duplicated(x, dx)
        )
        grad[i] = directional
        if i == firstindex(x)
            value = primal
        end
    end
    if isnothing(value)
        value = f(x)
    end
    return value, grad
end

function _value_and_jacobian(f, backend::AutoEnzyme, x::AbstractVector)
    mode = _jacobian_mode(backend)
    if mode isa EnzymeCore.ReverseMode
        annotated_f = _enzyme_annotate_f(f, backend, mode)
        jacs = Enzyme.jacobian(mode, annotated_f, x)
        value = f(x)
        return value, reshape(first(jacs), length(value), length(x))
    end

    withprimal_mode = _jacobian_withprimal_mode(backend)
    value = nothing
    J = nothing
    for i in eachindex(x)
        dx = zero(x)
        dx[i] = one(eltype(x))
        directional, primal = Enzyme.autodiff(
            withprimal_mode,
            _enzyme_annotate_f(f, backend, withprimal_mode),
            Enzyme.Duplicated(x, dx),
        )
        if i == firstindex(x)
            value = primal isa AbstractArray ? copy(primal) : primal
            J = Matrix{eltype(directional)}(undef, length(directional), length(x))
        end
        J[:, i] .= directional
    end
    if isnothing(value)
        value = f(x)
        J = Matrix{eltype(value)}(undef, length(value), 0)
    end
    return value, J
end

end
