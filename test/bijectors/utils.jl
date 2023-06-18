# Allows us to run `ChangesOfVariables.test_with_logabsdet_jacobian`
include(joinpath(dirname(pathof(ChangesOfVariables)), "..", "test", "getjacobian.jl"))

test_bijector(b, x; kwargs...) = test_bijector(b, x, getjacobian; kwargs...)

# TODO: Should we move this into `src/`?
function test_bijector(
    b,
    x,
    getjacobian;
    y=nothing,
    logjac=nothing,
    test_not_identity=isnothing(y) && isnothing(logjac),
    test_types=false,
    changes_of_variables_test=true,
    inverse_functions_test=true,
    test_sizes=true,
    compare=isapprox,
    kwargs...,
)
    # Ensure that everything is type-stable.
    ib = @inferred inverse(b)
    logjac_test = @inferred logabsdetjac(b, x)
    res = @inferred with_logabsdet_jacobian(b, x)

    y_test = @inferred b(x)
    ilogjac_test =
        !isnothing(y) ? @inferred(logabsdetjac(ib, y)) : @inferred(logabsdetjac(ib, y_test))
    ires = if !isnothing(y)
        @inferred(with_logabsdet_jacobian(inverse(b), y))
    else
        @inferred(with_logabsdet_jacobian(inverse(b), y_test))
    end

    if test_sizes
        @test Bijectors.output_size(b, size(x)) == size(y_test)
        @test Bijectors.output_size(ib, size(y_test)) == size(x)
    end

    # ChangesOfVariables.jl
    # For non-bijective transformations, these tests always fail since determinant of
    # the Jacobian is zero. Hence we allow the caller to disable them if necessary.
    if changes_of_variables_test
        ChangesOfVariables.test_with_logabsdet_jacobian(
            b, x, getjacobian; compare=compare, kwargs...
        )
        ChangesOfVariables.test_with_logabsdet_jacobian(
            ib, isnothing(y) ? y_test : y, getjacobian; compare=compare, kwargs...
        )
    end

    # InverseFunctions.jl
    if inverse_functions_test
        InverseFunctions.test_inverse(b, x; compare, kwargs...)
        InverseFunctions.test_inverse(
            ib, isnothing(y) ? y_test : y; compare=compare, kwargs...
        )
    end

    # Always want the following to hold
    @test compare(ires[1], x; kwargs...)
    @test compare(ires[2], -logjac_test; kwargs...)

    # Verify values.
    if !isnothing(y)
        @test compare(y_test, y; kwargs...)
        @test compare((@inferred ib(y)), x; kwargs...) # inverse
        @test compare(res[1], y; kwargs...)                 # forward using `forward`
    end

    if !isnothing(logjac)
        # We've already checked `ires[2]` against `res[2]`, so if `res[2]` is correct, then so is `ires[2]`.
        @test compare(logjac_test, logjac; kwargs...) # logjac forward
        @test compare(res[2], logjac; kwargs...) # logjac using `forward`
    end

    # Useful for testing when you don't know the true outputs but know that
    # `b` is definitively not identity.
    if test_not_identity
        @test y_test ≠ x
        @test logjac_test ≠ zero(eltype(x))
        @test res[2] ≠ zero(eltype(x))
    end

    if test_types
        @test typeof(first(res)) === typeof(x)
        @test typeof(res) === typeof(ires)
        @test typeof(y_test) === typeof(x)
        @test typeof(logjac_test) === typeof(ilogjac_test)
    end
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

# Check if `Functors.functor` works properly
function test_functor(x, xs)
    _xs, re = Functors.functor(x)
    @test x == re(_xs)
    @test _xs == xs
end

function test_bijector_parameter_gradient(b::Bijectors.Transform, x, y=b(x))
    args, re = Functors.functor(b)
    recon(k, param) = re(merge(args, NamedTuple{(k,)}((param,))))

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
