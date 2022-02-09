# TODO: Do we really need this?
Exp() = elementwise(exp)
Log() = elementwise(log)

transform!(b::Union{Elementwise{typeof(log)}, Elementwise{typeof(exp)}}, x, y) = broadcast!(b.x, y, x)

logabsdetjac(b::typeof(exp), x::Real) = x
logabsdetjac(b::Elementwise{typeof(exp)}, x) = sum(x)

logabsdetjac(b::typeof(log), x::Real) = -log(x)
logabsdetjac(b::Elementwise{typeof(log)}, x) = -sum(log, x)

function forward(b::typeof(exp), x::Real)
    y = b(x)
    return y, x
end
function forward(b::Elementwise{typeof(exp)}, x)
    y = b(x)
    return y, sum(x)
end

function forward(b::typeof(log), y::Real)
    x = transform(b, y)
    return x, -x
end
function forward(b::Elementwise{typeof(log)}, y)
    x = transform(b, y)
    return x, -sum(x)
end
