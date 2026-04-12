module BijectorsEnzymeExt

import Bijectors: _value_and_gradient, _value_and_jacobian
import ADTypes: AutoEnzyme
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore

function _enzyme_mode(backend::AutoEnzyme)
    return Enzyme.set_runtime_activity(backend.mode)
end

function _enzyme_annotate_f(f, ::AutoEnzyme{M,A}) where {M,A}
    if A <: EnzymeCore.Const
        return Enzyme.Const(f)
    else
        return f
    end
end

function _value_and_gradient(f, backend::AutoEnzyme, x::AbstractVector)
    mode = _enzyme_mode(backend)
    annotated_f = _enzyme_annotate_f(f, backend)
    grads = Enzyme.gradient(mode, annotated_f, x)
    return f(x), first(grads)
end

function _value_and_jacobian(f, backend::AutoEnzyme, x::AbstractVector)
    mode = _enzyme_mode(backend)
    annotated_f = _enzyme_annotate_f(f, backend)
    jacs = Enzyme.jacobian(mode, annotated_f, x)
    return f(x), first(jacs)
end

end
