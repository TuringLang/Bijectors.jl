import Bijectors: AbstractBatch


function test_bijector_single(
    b::Bijector,
    x_true,
    y_true,
    logjac_true;
    isequal = true,
    tol = 1e-6
)
    ib = @inferred inverse(b)
    y = @inferred b(x_true)
    logjac = @inferred logabsdetjac(b, x_true)
    ilogjac = @inferred logabsdetjac(ib, y_true)
    res = @inferred with_logabsdet_jacobian(b, x_true)

    # If `isequal` is false, then we use the computed `y`,
    # but if it's true, we use the true `y`.
    ires = isequal ? @inferred(with_logabsdet_jacobian(inverse(b), y_true)) : @inferred(with_logabsdet_jacobian(inverse(b), y))

    # Always want the following to hold
    @test ires[1] ≈ x_true atol=tol
    @test ires[2] ≈ -logjac atol=tol

    if isequal
        @test y ≈ y_true atol=tol                      # forward
        @test (@inferred ib(y_true)) ≈ x_true atol=tol # inverse
        @test logjac ≈ logjac_true                     # logjac forward
        @test res[1] ≈ y_true atol=tol                 # forward using `forward`
        @test res[2] ≈ logjac_true atol=tol  # logjac using `forward`
    else
        @test y ≠ y_true                          # forward
        @test (@inferred ib(y)) ≈ x_true atol=tol # inverse
        @test logjac ≠ logjac_true                # logjac forward
        @test res[1] ≠ y_true                     # forward using `forward`
        @test res[2] ≠ logjac_true      # logjac using `forward`
    end
end

function test_bijector_batch(
    b::Bijector,
    xs_true::AbstractBatch,
    ys_true::AbstractBatch,
    logjacs_true;
    isequal = true,
    tol = 1e-6
)
    ib = @inferred inverse(b)
    ys = @inferred b(xs_true)
    logjacs = @inferred logabsdetjac(b, xs_true)
    res = @inferred with_logabsdet_jacobian(b, xs_true)
    # If `isequal` is false, then we use the computed `y`,
    # but if it's true, we use the true `y`.
    ires = isequal ? @inferred(with_logabsdet_jacobian(inverse(b), ys_true)) : @inferred(with_logabsdet_jacobian(inverse(b), ys))

    # always want the following to hold
    @test ys isa typeof(ys_true)
    @test logjacs isa typeof(logjacs_true)
    @test mean(abs, ires[1] - xs_true) ≤ tol
    @test mean(abs, ires[2] + logjacs) ≤ tol

    if isequal
        @test mean(abs, ys - ys_true) ≤ tol                     # forward
        @test mean(abs, (ib(ys_true)) - xs_true) ≤ tol          # inverse
        @test mean(abs, logjacs - logjacs_true) ≤ tol           # logjac forward
        @test mean(abs, res[1] - ys_true) ≤ tol                 # forward using `forward`
        @test mean(abs, res[2] - logjacs_true) ≤ tol  # logjac `forward`
        @test mean(abs, ires[2] + logjacs_true) ≤ tol # inverse logjac `forward`
    else
        # Don't want the following to be equal to their "true" values
        @test mean(norm, ys - ys_true) > tol           # forward
        @test mean(abs, logjacs - logjacs_true) > tol # logjac forward
        @test mean(abs, res[1] - ys_true) > tol       # forward using `forward`

        # Still want the following to be equal to the COMPUTED values
        @test mean(abs, ib(ys) - xs_true) ≤ tol           # inverse
        @test mean(abs, res[2] - logjacs) ≤ tol # logjac forward using `forward`
    end
end

"""
    test_bijector(b::Bijector, xs::Array, ys::Array, logjacs::Array; kwargs...)

Tests the bijector `b` on the inputs `xs` against the, optionally, provided `ys`
and `logjacs`.

If `ys` and `logjacs` are NOT provided, `isequal` will be set to `false` and
`ys` and `logjacs` will be set to `zeros`. These `ys` and `logjacs` will be
treated as "counter-examples", i.e. values NOT to match.

# Arguments
- `b::Bijector`: the bijector to test
- `xs`: inputs (has to be several!!!)(has to be several, i.e. a batch!!!) to test
- `ys`: outputs (has to be several, i.e. a batch!!!) to test against
- `logjacs`: `logabsdetjac` outputs (has to be several!!!)(has to be several, i.e. 
  a batch!!!) to test against

# Keywords
- `isequal = true`: if `false`, it will be assumed that the given values are
  provided as "counter-examples" in the sense that the inputs `xs` should NOT map
  to the given outputs. This is useful in cases where one might not know the expected
  output, but still wants to test that the evaluation, etc. works.
  This is set to `true` by default if `ys` and `logjacs` are not provided.
- `tol = 1e-6`: the absolute tolerance used for the checks. This is also used to check
  arrays where we check that the L1-norm is sufficiently small.
"""
function test_bijector(
    b::Bijector,
    xs_true::AbstractBatch,
    ys_true::AbstractBatch,
    logjacs_true::AbstractBatch;
    kwargs...
)
    ib = inverse(b)

    # Batch
    test_bijector_arrays(b, xs_true, ys_true, logjacs_true; kwargs...)

    # Test `logabsdetjac` against jacobians
    test_logabsdetjac(b, xs_true)

    if Bijectors.isinvertible(b)
        ib = inv(b)
        test_logabsdetjac(ib, ys_true)
    end
    
    for (x_true, y_true, logjac_true) in zip(xs_true, ys_true, logjacs_true)
        # Test validity of single input.
        test_bijector_single(b, x_true, y_true, logjac_true; kwargs...)

        # Test AD wrt. inputs.
        test_ad(x -> sum(b(x)), x_true)
        test_ad(x -> logabsdetjac(b, x), x_true)

        if Bijectors.isinvertible(b)
            y = b(x_true)
            test_ad(x -> sum(ib(x)), y)
        end
    end

    # Test AD wrt. parameters.
    test_bijector_parameter_gradient(b, xs_true[1], ys_true[1])

    # Test validity of collection of inputs.
    test_bijector_batch(b, xs_true, ys_true, logjacs_true; kwargs...)

    # AD testing for batch.
    f, arg = make_gradient_function(x -> sum(sum(b.(x))), xs_true)
    test_ad(f, arg)
    f, arg = make_gradient_function(x -> sum(logabsdetjac.(b, x)), xs_true)
    test_ad(f, arg)

    if Bijectors.isinvertible(b)
        ys = b.(xs_true)
        f, arg = make_gradient_function(y -> sum(sum(ib.(y))), ys)
        test_ad(f, arg)
    end
end

function make_gradient_function(f, xs::ArrayBatch)
    s = size(Bijectors.value(xs))

function test_bijector(
    b::Bijector{1},
    xs_true::AbstractMatrix{<:Real},
    ys_true::AbstractMatrix{<:Real},
    logjacs_true::AbstractVector{<:Real};
    kwargs...
)
    ib = inverse(b)

    return g, vec(Bijectors.value(xs))
end

function make_gradient_function(f, xs::VectorBatch{<:AbstractArray{<:Real}})
    xs_new = vcat(map(vec, Bijectors.value(xs)))
    n = length(xs_new)

    s = size(Bijectors.value(xs[1]))
    stride = n ÷ length(xs)

    function g(x)
        x_vec = map(1:stride:n) do i
            reshape(x[i:i + stride - 1], s)
        end

        x_batch = Bijectors.reconstruct(xs, x_vec)
        return f(x_batch)
    end

    return g, xs_new
end

make_jacobian_function(f, xs::AbstractVector) = f, xs
function make_jacobian_function(f, xs::AbstractArray)
    xs_new = vec(xs)
    s = size(xs)

    function g(x)
        return vec(f(reshape(x, s)))
    end

    return g, xs_new
end

function test_logabsdetjac(b::Transform, xs::Batch{<:Any, <:AbstractArray}; tol=1e-6)
    f, _ = make_jacobian_function(b, xs[1])
    logjac_ad = map(xs) do x
        first(logabsdet(ForwardDiff.jacobian(f, x)))
    end

    @test mean(collect(logabsdetjac.(b, xs)) - logjac_ad) ≤ tol
end

function test_logabsdetjac(b::Transform, xs::Batch{<:Any, <:Real}; tol=1e-6)
    logjac_ad = map(xs) do x
        log(abs(ForwardDiff.derivative(b, x)))
    end
    @test mean(collect(logabsdetjac.(b, xs)) - logjac_ad) ≤ tol
end

# Check if `Functors.functor` works properly
function test_functor(x, xs)
    _xs, re = Functors.functor(x)
    @test x == re(_xs)
    @test _xs == xs
end

function test_bijector_parameter_gradient(b::Transform, x, y = b(x))
    args, re = Functors.functor(b)
    recon(k, param) = re(merge(args, NamedTuple{(k, )}((param, ))))

    # Compute the gradient wrt. one argument at the time.
    for (k, v) in pairs(args)
        test_ad(p -> sum(transform(recon(k, p), x)), v)
        test_ad(p -> logabsdetjac(recon(k, p), x), v)

        if Bijectors.isinvertible(b)
            test_ad(p -> sum(transform(inv(recon(k, p)), y)), v)
            test_ad(p -> logabsdetjac(inv(recon(k, p)), y), v)
        end
    end
end
