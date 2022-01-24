transform_single!(b::Union{Elementwise{typeof(log)}, Elementwise{typeof(exp)}}, x, y) = broadcast!(b.x, y, x)

transform_multiple(b::Union{typeof(log), typeof(exp)}, x) = b.(x)
transform_multiple!(b::Union{typeof(log), typeof(exp)}, x, y) = broadcast!(b, y, x)

logabsdetjac_single(b::typeof(exp), x::Real) = x
logabsdetjac_single(b::Elementwise{typeof(exp)}, x) = sum(x)

logabsdetjac_single(b::typeof(log), x::Real) = -log(x)
logabsdetjac_single(b::Elementwise{typeof(log)}, x) = -sum(log, x)

logabsdetjac_multiple(b::typeof(exp), xs) = xs
logabsdetjac_multiple(b::Elementwise{typeof(exp)}, xs) = map(sum, xs)

logabsdetjac_multiple(b::typeof(log), xs) = -map(log, xs)
logabsdetjac_multiple(b::Elementwise{typeof(log)}, xs) = -map(sum ∘ log, xs)

function forward_single(b::typeof(exp), x::Real)
    y = b(x)
    return y, x
end
function forward_single(b::Elementwise{typeof(exp)}, x)
    y = b(x)
    return y, sum(x)
end

function forward_multiple(b::typeof(exp), xs::AbstractBatch{<:Real})
    ys = transform(b, xs)
    return ys, xs
end
function forward_multiple!(
    b::typeof(exp),
    xs::AbstractBatch{<:Real},
    ys::AbstractBatch{<:Real},
    logjacs::AbstractBatch{<:Real}
)
    transform!(b, xs, ys)
    logjacs += xs
    return ys, logjacs
end
function forward_multiple(b::Elementwise{typeof(exp)}, xs)
    ys = transform(b, xs)
    return ys, map(sum, xs)
end
function forward_multiple!(b::Elementwise{typeof(exp)}, xs, ys, logjacs)
    # Do this before `transform!` in case `xs === ys`.
    logjacs += map(sum, xs)
    transform!(b, xs, ys)
    return ys, logjacs
end

function forward_single(b::typeof(log), y::Real)
    x = transform(b, y)
    return x, -x
end
function forward_single(b::Elementwise{typeof(log)}, y)
    x = transform(b, y)
    return x, -sum(x)
end

function forward_multiple(b::typeof(log), ys::AbstractBatch{<:Real})
    xs = transform(b, ys)
    return xs, -xs
end
function forward_multiple!(
    b::typeof(log),
    ys::AbstractBatch{<:Real},
    xs::AbstractBatch{<:Real},
    logjacs::AbstractBatch{<:Real}
)
    transform!(b, ys, xs)
    logjacs -= xs
    return xs, logjacs
end
function forward_multiple(b::Elementwise{typeof(log)}, ys)
    xs = transform(b, ys)
    return xs, -map(sum, xs)
end
function forward_multiple!(b::Elementwise{typeof(log)}, ys, xs, logjacs)
    transform!(b, ys, xs)
    logjacs -= map(sum, xs)
    return xs, logjacs
end
