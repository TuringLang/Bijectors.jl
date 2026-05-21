function transform!(b::Union{Elementwise{typeof(log)},Elementwise{typeof(exp)}}, x, y)
    return broadcast!(b.x, y, x)
end

logabsdetjac(b::typeof(exp), x::Real) = x
logabsdetjac(b::Elementwise{typeof(exp)}, x) = sum(x)

logabsdetjac(b::typeof(log), x::Real) = -log(x)
logabsdetjac(b::Elementwise{typeof(log)}, x) = -sum(log, x)

is_monotonically_increasing(::typeof(exp)) = true
is_monotonically_increasing(::typeof(log)) = true
