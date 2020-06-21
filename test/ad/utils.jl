# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")

# Struct of distribution, corresponding parameters, and a sample.
struct DistSpec{VF<:VariateForm,VS<:ValueSupport,F,T,X,G}
    name::Symbol
    f::F
    "Distribution parameters."
    θ::T
    "Sample."
    x::X
    "Transformation of sample `x`."
    xtrans::G
end

function DistSpec(f, θ, x, xtrans=nothing)
    name = f isa Distribution ? nameof(typeof(f)) : nameof(typeof(f(θ...)))
    return DistSpec(name, f, θ, x, xtrans)
end

function DistSpec(name::Symbol, f, θ, x, xtrans=nothing)
    F = f isa Distribution ? typeof(f) : typeof(f(θ...))
    VF = Distributions.variate_form(F)
    VS = Distributions.value_support(F)
    return DistSpec{VF,VS,typeof(f),typeof(θ),typeof(x),typeof(xtrans)}(
        name, f, θ, x, xtrans
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
function unpack_offset(x, offset, original::Diagonal)
    newoffset = offset + size(original, 1)
    val = Diagonal(x[(offset + 1):newoffset])
    return val, newoffset
end
function unpack_offset(x, offset, original::AbstractVector{<:AbstractMatrix})
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

    if isempty(θ)
        # In this case we can only test the gradient with respect to `x`
        xtest = vectorize(x)
        ftest = let xorig=x
            x -> f_allargs(unpack(x, (1,), xorig)...)
        end
        test_ad(ftest, xtest; kwargs...)
    else
        # For all combinations of distribution parameters `θ`
        for inds in combinations(2:(length(θ) + 1))
            # Test only distribution parameters
            xtest = mapreduce(vcat, inds) do i
                vectorize(θ[i - 1])
            end
            ftest = let xorig=x, θorig=θ, inds=inds
                x -> f_allargs(unpack(x, inds, xorig, θorig...)...)
            end
            test_ad(ftest, xtest; kwargs...)

            # Test derivative with respect to location `x` as well
            # if the distribution is continuous
            if Distributions.value_support(typeof(dist)) === Continuous
                xtest = vcat(vectorize(x), xtest)
                push!(inds, 1)
                ftest = let xorig=x, θorig=θ, inds=inds
                    x -> f_allargs(unpack(x, inds, xorig, θorig...)...)
                end
                test_ad(ftest, xtest; kwargs...)
            end
        end
    end
end

function test_ad(f, x; rtol = 1e-6, atol = 1e-6)
    finitediff = FiniteDiff.finite_difference_gradient(f, x)

    if AD == "All" || AD == "ForwardDiff_Tracker"
        tracker = Tracker.data(Tracker.gradient(f, x)[1])
        @test tracker ≈ finitediff rtol=rtol atol=atol

        forward = ForwardDiff.gradient(f, x)
        @test forward ≈ finitediff rtol=rtol atol=atol
    end

    if AD == "All" || AD == "Zygote"
        zygote = Zygote.gradient(f, x)[1]
        @test zygote ≈ finitediff rtol=rtol atol=atol
    end

    if AD == "All" || AD == "ReverseDiff"
        reversediff = ReverseDiff.gradient(f, x)
        @test reversediff ≈ finitediff rtol=rtol atol=atol
    end

    return
end
