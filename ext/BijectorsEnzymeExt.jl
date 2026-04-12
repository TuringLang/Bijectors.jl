module BijectorsEnzymeExt

import Bijectors: _value_and_gradient, _value_and_jacobian
import ADTypes: AutoEnzyme
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore

# When mode is unspecified, fall back to Reverse.
function _enzyme_mode(::AutoEnzyme{Nothing})
    return Enzyme.set_runtime_activity(Enzyme.Reverse)
end
function _enzyme_mode(backend::AutoEnzyme)
    return Enzyme.set_runtime_activity(backend.mode)
end

# Returns f annotated as requested, or bare f when annotation is Nothing.
function _enzyme_annotate_f(f, ::AutoEnzyme{M,A}) where {M,A}
    if A <: EnzymeCore.Const
        return Enzyme.Const(f)
    else
        return f
    end
end

# For reverse mode (explicit or auto/nothing), use ReverseWithPrimal so value and
# gradient are computed in a single autodiff call rather than evaluating f twice.
function _value_and_gradient(
    f,
    backend::Union{AutoEnzyme{Nothing},AutoEnzyme{<:EnzymeCore.ReverseMode}},
    x::AbstractVector,
)
    annotated_f = _enzyme_annotate_f(f, backend)
    dx = zero(x)
    _, val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        annotated_f,
        Enzyme.Active,
        Enzyme.Duplicated(x, dx),
    )
    return val, dx
end

# For forward mode the gradient already requires O(n) JVPs; one extra f(x) evaluation
# is negligible.
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
