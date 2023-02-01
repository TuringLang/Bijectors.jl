isinvertible(cb::ComposedFunction) = isinvertible(cb.inner) && isinvertible(cb.outer)
isclosedform(cb::ComposedFunction) = isclosedform(cb.inner) && isclosedform(cb.outer)

transform(cb::ComposedFunction, x) = transform(cb.outer, transform(cb.inner, x))

function transform!(cb::ComposedFunction, x, y)
    transform!(cb.inner, x, y)
    return transform!(cb.outer, y, y)
end

function logabsdetjac(cb::ComposedFunction, x)
    y, logjac = with_logabsdet_jacobian(cb.inner, x)
    return logabsdetjac(cb.outer, y) + logjac
end

function logabsdetjac!(cb::ComposedFunction, x, logjac)
    y = similar(x)
    logjac = last(with_logabsdet_jacobian!(cb.inner, x, y, logjac))
    return logabsdetjac!(cb.outer, y, y, logjac)
end

function with_logabsdet_jacobian!(cb::ComposedFunction, x, y, logjac)
    logjac = last(with_logabsdet_jacobian!(cb.inner, x, y, logjac))
    return with_logabsdet_jacobian!(cb.outer, y, y, logjac)
end
