function test_bijector_reals(
    b::Bijector{0},
    x_true::Real,
    y_true::Real,
    logjac_true::Real;
    isequal = true,
    tol = 1e-6
)
    ib = @inferred inv(b)
    y = @inferred b(x_true)
    logjac = @inferred logabsdetjac(b, x_true)
    ilogjac = @inferred logabsdetjac(ib, y_true)
    res = @inferred forward(b, x_true)

    # If `isequal` is false, then we use the computed `y`,
    # but if it's true, we use the true `y`.
    ires = isequal ? @inferred(forward(inv(b), y_true)) : @inferred(forward(inv(b), y))

    # Always want the following to hold
    @test ires.rv ≈ x_true atol=tol
    @test ires.logabsdetjac ≈ -logjac atol=tol

    if isequal
        @test y ≈ y_true atol=tol                      # forward
        @test (@inferred ib(y_true)) ≈ x_true atol=tol # inverse
        @test logjac ≈ logjac_true                     # logjac forward
        @test res.rv ≈ y_true atol=tol                 # forward using `forward`
        @test res.logabsdetjac ≈ logjac_true atol=tol  # logjac using `forward`
    else
        @test y ≠ y_true                          # forward
        @test (@inferred ib(y)) ≈ x_true atol=tol # inverse
        @test logjac ≠ logjac_true                # logjac forward
        @test res.rv ≠ y_true                     # forward using `forward`
        @test res.logabsdetjac ≠ logjac_true      # logjac using `forward`
    end
end

function test_bijector_arrays(
    b::Bijector,
    xs_true::AbstractArray{<:Real},
    ys_true::AbstractArray{<:Real},
    logjacs_true::Union{Real, AbstractArray{<:Real}};
    isequal = true,
    tol = 1e-6
)
    ib = @inferred inv(b)
    ys = @inferred b(xs_true)
    logjacs = @inferred logabsdetjac(b, xs_true)
    res = @inferred forward(b, xs_true)
    # If `isequal` is false, then we use the computed `y`,
    # but if it's true, we use the true `y`.
    ires = isequal ? @inferred(forward(inv(b), ys_true)) : @inferred(forward(inv(b), ys))

    # always want the following to hold
    @test ys isa typeof(ys_true)
    @test logjacs isa typeof(logjacs_true)
    @test mean(abs, ires.rv - xs_true) ≤ tol
    @test mean(abs, ires.logabsdetjac + logjacs) ≤ tol

    if isequal
        @test mean(abs, ys - ys_true) ≤ tol                     # forward
        @test mean(abs, (ib(ys_true)) - xs_true) ≤ tol          # inverse
        @test mean(abs, logjacs - logjacs_true) ≤ tol           # logjac forward
        @test mean(abs, res.rv - ys_true) ≤ tol                 # forward using `forward`
        @test mean(abs, res.logabsdetjac - logjacs_true) ≤ tol  # logjac `forward`
        @test mean(abs, ires.logabsdetjac + logjacs_true) ≤ tol # inverse logjac `forward`
    else
        # Don't want the following to be equal to their "true" values
        @test mean(abs, ys - ys_true) > tol           # forward
        @test mean(abs, logjacs - logjacs_true) > tol # logjac forward
        @test mean(abs, res.rv - ys_true) > tol       # forward using `forward`

        # Still want the following to be equal to the COMPUTED values
        @test mean(abs, ib(ys) - xs_true) ≤ tol           # inverse
        @test mean(abs, res.logabsdetjac - logjacs) ≤ tol # logjac forward using `forward`
    end
end

"""
    test_bijector(b::Bijector, xs::Array; kwargs...)
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
function test_bijector(b::Bijector{0}, xs::AbstractVector{<:Real})
    return test_bijector(b, xs, zeros(length(xs)), zeros(length(xs)); isequal = false)
end

function test_bijector(b::Bijector{1}, xs::AbstractMatrix{<:Real})
    return test_bijector(b, xs, zeros(size(xs)), zeros(size(xs, 2)); isequal = false)
end

function test_bijector(
    b::Bijector{0},
    xs_true::AbstractVector{<:Real},
    ys_true::AbstractVector{<:Real},
    logjacs_true::AbstractVector{<:Real};
    kwargs...
)
    ib = inv(b)

    # Batch
    test_bijector_arrays(b, xs_true, ys_true, logjacs_true; kwargs...)

    # Test `logabsdetjac` against jacobians
    test_logabsdetjac(b, xs_true)
    test_logabsdetjac(b, ys_true)
    
    for (x_true, y_true, logjac_true) in zip(xs_true, ys_true, logjacs_true)
        test_bijector_reals(b, x_true, y_true, logjac_true; kwargs...)

        # Test AD
        if isclosedform(b)
            test_ad(x -> b(first(x)), [x_true, ])
        end

        if isclosedform(ib)
            y = b(x_true)
            test_ad(x -> ib(first(x)), [y, ])
        end

        test_ad(x -> logabsdetjac(b, first(x)), [x_true, ])
    end
end


function test_bijector(
    b::Bijector{1},
    xs_true::AbstractMatrix{<:Real},
    ys_true::AbstractMatrix{<:Real},
    logjacs_true::AbstractVector{<:Real};
    kwargs...
)
    ib = inv(b)

    # Batch
    test_bijector_arrays(b, xs_true, ys_true, logjacs_true; kwargs...)

    # Test `logabsdetjac` against jacobians
    test_logabsdetjac(b, xs_true)
    test_logabsdetjac(b, ys_true)
    
    for (x_true, y_true, logjac_true) in zip(eachcol(xs_true), eachcol(ys_true), logjacs_true)
        # HACK: collect to avoid dealing with sub-arrays and thus allowing us to compare the
        # type of the computed output to the "true" output.
        test_bijector_arrays(b, collect(x_true), collect(y_true), logjac_true; kwargs...)

        # Test AD
        if isclosedform(b)
            test_ad(x -> sum(b(x)), collect(x_true))
        end
        if isclosedform(ib)
            y = b(x_true)
            test_ad(x -> sum(ib(x)), y)
        end

        test_ad(x -> logabsdetjac(b, x), x_true)
    end
end

function test_logabsdetjac(b::Bijector{1}, xs::AbstractMatrix; tol=1e-6)
    if isclosedform(b)
        logjac_ad = [logabsdet(ForwardDiff.jacobian(b, x))[1] for x in eachcol(xs)]
        @test mean(logabsdetjac(b, xs) - logjac_ad) ≤ tol
    end
end

function test_logabsdetjac(b::Bijector{0}, xs::AbstractVector; tol=1e-6)
    if isclosedform(b)
        logjac_ad = [log(abs(ForwardDiff.derivative(b, x))) for x in xs]
        @test mean(logabsdetjac(b, xs) - logjac_ad) ≤ tol
    end
end
