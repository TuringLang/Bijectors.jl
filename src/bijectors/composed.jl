invertible(cb::ComposedFunction) = invertible(cb.inner) + invertible(cb.outer)
isclosedform(cb::ComposedFunction) = isclosedform(cb.inner) && isclosedform(cb.outer)

transform(cb::ComposedFunction, x) = transform(cb.outer, transform(cb.inner, x))

function transform!(cb::ComposedFunction, x, y)
    transform!(cb.inner, x, y)
    return transform!(cb.outer, y, y)
end

function logabsdetjac(cb::ComposedFunction, x)
    y, logjac = forward(cb.inner, x)
    return logabsdetjac(cb.outer, y) + logjac
end

function logabsdetjac!(cb::ComposedFunction, x, logjac)
    y = similar(x)
    forward!(cb.inner, x, y, logjac)
    return logabdetjac!(cb.outer, y, y, logjac)
end

function forward(cb::ComposedFunction, x)
    y1, logjac1 = forward(cb.inner, x)
    y2, logjac2 = forward(cb.outer, y1)
    return y2, logjac1 + logjac2
end

function forward!(cb::ComposedFunction, x, y, logjac)
    forward!(cb.inner, x, y, logjac)
    return forward!(cb.outer, y, y, logjac)
end
