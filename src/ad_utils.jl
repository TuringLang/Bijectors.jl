"""
    _value_and_gradient(f, backend::ADTypes.AbstractADType, x::AbstractVector)

Compute the value and gradient of scalar-valued `f` at `x`.
Returns `(f(x), gradient)` where `gradient` is a vector of the same size as `x`.

Implementations are provided by package extensions for each AD backend.
"""
function _value_and_gradient(f, backend::ADTypes.AbstractADType, x::AbstractVector)
    return error(
        "_value_and_gradient is not implemented for backend $(typeof(backend)). " *
        "Load the corresponding AD package to enable support.",
    )
end

"""
    _value_and_jacobian(f, backend::ADTypes.AbstractADType, x::AbstractVector)

Compute the value and Jacobian of `f` at `x`.
Returns `(f(x), jacobian)` where `jacobian` is a matrix of size
`(length(f(x)), length(x))`.

Implementations are provided by package extensions for each AD backend.
"""
function _value_and_jacobian(f, backend::ADTypes.AbstractADType, x::AbstractVector)
    return error(
        "_value_and_jacobian is not implemented for backend $(typeof(backend)). " *
        "Load the corresponding AD package to enable support.",
    )
end
