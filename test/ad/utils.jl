# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")

function test_ad(f, x, broken=(); rtol=1e-6, atol=1e-6, use_forwarddiff_as_truth=false)
    truth = if use_forwarddiff_as_truth
        truth = ForwardDiff.gradient(f, x)
    else
        FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]
    end

    if !use_forwarddiff_as_truth && (AD == "All" || AD == "ForwardDiff")
        if :ForwardDiff in broken
            @test_broken ForwardDiff.gradient(f, x) ≈ truth rtol = rtol atol = atol
        else
            @test ForwardDiff.gradient(f, x) ≈ truth rtol = rtol atol = atol
        end
    end

    if AD == "All" || AD == "Zygote"
        if :Zygote in broken
            @test_broken Zygote.gradient(f, x)[1] ≈ truth rtol = rtol atol = atol
        else
            ∇zygote = Zygote.gradient(f, x)[1]
            @test (all(truth .== 0) && ∇zygote === nothing) ||
                isapprox(∇zygote, truth; rtol=rtol, atol=atol)
        end
    end

    if AD == "All" || AD == "ReverseDiff"
        if :ReverseDiff in broken
            @test_broken ReverseDiff.gradient(f, x) ≈ truth rtol = rtol atol = atol
        else
            @test ReverseDiff.gradient(f, x) ≈ truth rtol = rtol atol = atol
        end
    end

    return nothing
end
