# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")

# Struct of distribution, corresponding parameters, and a sample.
struct DistSpec{VF<:VariateForm,VS<:ValueSupport,F,T,X,G,B<:Tuple}
    name::Symbol
    f::F
    "Distribution parameters."
    θ::T
    "Sample."
    x::X
    "Transformation of sample `x`."
    xtrans::G
    "Broken backends"
    broken::B
end

function DistSpec(f, θ, x, xtrans=nothing; broken=())
    name = f isa Distribution ? nameof(typeof(f)) : nameof(typeof(f(θ...)))
    return DistSpec(name, f, θ, x, xtrans; broken=broken)
end

function DistSpec(name::Symbol, f, θ, x, xtrans=nothing; broken=())
    F = f isa Distribution ? typeof(f) : typeof(f(θ...))
    VF = Distributions.variate_form(F)
    VS = Distributions.value_support(F)
    return DistSpec{VF,VS,typeof(f),typeof(θ),typeof(x),typeof(xtrans),typeof(broken)}(
        name, f, θ, x, xtrans, broken,
    )
end

Distributions.variate_form(::Type{<:DistSpec{VF}}) where VF = VF
Distributions.value_support(::Type{<:DistSpec{VF,VS}}) where {VF,VS} = VS

# Auxiliary method for vectorizing parameters and samples
vectorize(v::Number) = [v]
vectorize(v::Diagonal) = v.diag
vectorize(v::AbstractVector{<:AbstractMatrix}) = mapreduce(vectorize, vcat, v)
vectorize(v) = vec(v)

"""
    unpack(x, inds, original...)

Return a tuple of unpacked parameters and samples in vector `x`.

Here `original` are the original full set of parameters and samples, and
`inds` contains the indices of the original parameters and samples for which
a possibly different value is given in `x`. If no value is provided in `x`,
the original value of the parameter is returned. The values are returned
in the same order as the original parameters.
"""
function unpack(x, inds, original...)
    offset = 0
    newvals = ntuple(length(original)) do i
        if i in inds
            v, offset = unpack_offset(x, offset, original[i])
        else
            v = original[i]
        end
        return v
    end
    offset == length(x) || throw(ArgumentError())

    return newvals
end

# Auxiliary methods for unpacking numbers and arrays
function unpack_offset(x, offset, original::Number)
    newoffset = offset + 1
    val = x[newoffset]
    return val, newoffset
end
function unpack_offset(x, offset, original::AbstractArray)
    newoffset = offset + length(original)
    val = reshape(x[(offset + 1):newoffset], size(original))
    return val, newoffset
end
function unpack_offset(x, offset, original::AbstractArray{<:AbstractArray})
    newoffset = offset
    val = map(original) do orig
        out, newoffset = unpack_offset(x, newoffset, orig)
        return out
    end
    return val, newoffset
end

# Run AD tests of a
function test_ad(dist::DistSpec; kwargs...)
    @info "Testing: $(dist.name)"

    f = dist.f
    θ = dist.θ
    x = dist.x
    g = dist.xtrans
    broken = dist.broken

    # Test links
    d = f(θ...)
    g_x = g === nothing ? x : g(x)
    @test invlink(d, link(d, g_x)) ≈ g_x

    # Create function with all possible arguments
    f_allargs = let f=f, g=g
        function (x, θ...)
            dist = f(θ...)
            xtilde = g === nothing ? x : g(x)
            y = link(dist, xtilde)
            result = logpdf_with_trans(dist, invlink(dist, y), true)
            return sum(result)
        end
    end

    # For all combinations of distribution parameters `θ`
    for inds in powerset(2:(length(θ) + 1))
        # Test only distribution parameters
        if !isempty(inds)
            xtest = mapreduce(vcat, inds) do i
                vectorize(θ[i - 1])
            end
            f_test = let xorig=x, θorig=θ, inds=inds
                x -> f_allargs(unpack(x, inds, xorig, θorig...)...)
            end
            test_ad(f_test, xtest, broken; kwargs...)
        end

        # Test derivative with respect to location `x` as well
        # if the distribution is continuous
        if Distributions.value_support(typeof(dist)) === Continuous
            xtest = isempty(inds) ? vectorize(x) : vcat(vectorize(x), xtest)
            push!(inds, 1)
            f_test = let xorig=x, θorig=θ, inds=inds
                x -> f_allargs(unpack(x, inds, xorig, θorig...)...)
            end
            test_ad(f_test, xtest, broken; kwargs...)
        end
    end
end

function test_ad(_f, _x, broken = (); rtol = 1e-6, atol = 1e-6)
    f = _x isa Real ? _f ∘ first : _f
    x = [_x;]

    finitediff = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]

    if AD == "All" || AD == "Tracker"
        if :Tracker in broken
            @test_broken Tracker.data(Tracker.gradient(f, x)[1]) ≈ finitediff rtol=rtol atol=atol
        else
            ∇tracker = Tracker.gradient(f, x)[1]
            @test Tracker.data(∇tracker) ≈ finitediff rtol=rtol atol=atol
            @test Tracker.istracked(∇tracker)
        end
    end

    if AD == "All" || AD == "ForwardDiff"
        if :ForwardDiff in broken
            @test_broken ForwardDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        else
            @test ForwardDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        end
    end

    if AD == "All" || AD == "Zygote"
        if :Zygote in broken
            @test_broken Zygote.gradient(f, x)[1] ≈ finitediff rtol=rtol atol=atol
        else
            ∇zygote = Zygote.gradient(f, x)[1]
            @test (all(finitediff .== 0) && ∇zygote === nothing) || isapprox(∇zygote, finitediff, rtol=rtol, atol=atol)
        end
    end

    if AD == "All" || AD == "ReverseDiff"
        if :ReverseDiff in broken
            @test_broken ReverseDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        else
            @test ReverseDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        end
    end

    return
end
