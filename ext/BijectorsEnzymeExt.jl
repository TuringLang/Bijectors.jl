module BijectorsEnzymeExt

import Bijectors: _value_and_gradient, _value_and_jacobian
import ADTypes: AutoEnzyme
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore

const DuplicatedFunctionAnnotations = Union{
    EnzymeCore.Duplicated,EnzymeCore.DuplicatedNoNeed,EnzymeCore.MixedDuplicated
}

function _annotate_function(f, backend::AutoEnzyme, mode)
    annotation = typeof(backend).parameters[2]
    if annotation === Nothing
        return f
    elseif annotation <: EnzymeCore.Const
        return Enzyme.Const(f)
    elseif annotation <: DuplicatedFunctionAnnotations
        if Enzyme.guess_activity(typeof(f), mode) <: EnzymeCore.Const
            return Enzyme.Const(f)
        else
            # Enzyme's sugar APIs only preserve function shadows for `Duplicated`,
            # so normalize the duplicated-like annotations here.
            return Enzyme.Duplicated(f, Enzyme.make_zero(f))
        end
    else
        throw(ArgumentError("unsupported Enzyme function annotation $annotation"))
    end
end

function _value_and_gradient(
    f,
    backend::Union{AutoEnzyme{Nothing},AutoEnzyme{<:EnzymeCore.ReverseMode}},
    x::AbstractVector,
)
    mode = if backend isa AutoEnzyme{Nothing}
        Enzyme.ReverseWithPrimal
    else
        Enzyme.WithPrimal(backend.mode)
    end
    annotated_f = _annotate_function(f, backend, mode)
    dx = zero(x)
    _, val = Enzyme.autodiff(mode, annotated_f, Enzyme.Active, Enzyme.Duplicated(x, dx))
    return val, dx
end

function _value_and_gradient(
    f, backend::AutoEnzyme{<:EnzymeCore.ForwardMode}, x::AbstractVector
)
    mode = Enzyme.WithPrimal(backend.mode)
    annotated_f = _annotate_function(f, backend, mode)
    grad = zero(x)
    value = f(x)
    for i in eachindex(x)
        dx = zero(x)
        dx[i] = one(eltype(x))
        directional, primal = Enzyme.autodiff(mode, annotated_f, Enzyme.Duplicated(x, dx))
        grad[i] = directional
        if i == firstindex(x)
            value = primal
        end
    end
    return value, grad
end

function _value_and_jacobian(
    f, backend::AutoEnzyme{<:EnzymeCore.ReverseMode}, x::AbstractVector
)
    value = f(x)
    if isempty(x)
        return value, Matrix{eltype(value)}(undef, length(value), 0)
    end
    annotated_f = _annotate_function(f, backend, backend.mode)
    jacobian = only(Enzyme.jacobian(backend.mode, annotated_f, x))
    return value, reshape(jacobian, length(value), length(x))
end

function _value_and_jacobian(f, ::AutoEnzyme{Nothing}, x::AbstractVector)
    return _value_and_jacobian(f, AutoEnzyme(; mode=Enzyme.Forward), x)
end

function _value_and_jacobian(
    f, backend::AutoEnzyme{<:EnzymeCore.ForwardMode}, x::AbstractVector
)
    mode = Enzyme.WithPrimal(backend.mode)
    annotated_f = _annotate_function(f, backend, mode)
    value = f(x)
    J = nothing
    for i in eachindex(x)
        dx = zero(x)
        dx[i] = one(eltype(x))
        directional, primal = Enzyme.autodiff(mode, annotated_f, Enzyme.Duplicated(x, dx))
        if i == firstindex(x)
            value = primal isa AbstractArray ? copy(primal) : primal
            J = Matrix{eltype(directional)}(undef, length(directional), length(x))
        end
        J[:, i] .= directional
    end
    if isnothing(J)
        J = Matrix{eltype(value)}(undef, length(value), 0)
    end
    return value, J
end

end
