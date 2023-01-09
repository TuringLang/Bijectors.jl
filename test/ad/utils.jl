# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")

function test_ad(f, x, broken = (); rtol = 1e-6, atol = 1e-6)
    finitediff = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]

    if AD == "All" || AD == "Tracker"
        if :Tracker in broken
            @test_broken Tracker.data(Tracker.gradient(f, x)[1]) ≈ finitediff rtol=rtol atol=atol
        else
            @test Tracker.data(Tracker.gradient(f, x)[1]) ≈ finitediff rtol=rtol atol=atol
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
            @test Zygote.gradient(f, x)[1] ≈ finitediff rtol=rtol atol=atol
        end
    end

    if AD == "All" || AD == "ReverseDiff"
        if :ReverseDiff in broken
            @test_broken ReverseDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        else
            @test ReverseDiff.gradient(f, x) ≈ finitediff rtol=rtol atol=atol
        end
    end

    if AD == "All" || All == "Enzyme"
        # `broken` keyword to `@test` requires Julia >= 1.7 
        if :EnzymeReverse in broken
            @test Enzyme.gradient(Enzyme.Forward, f, z) ≈ finitediff rtol=rtol atol=atol
            @test_broken Enzyme.gradient(Enzyme.Reverse, f, z) ≈ finitediff rtol=rtol atol=atol
        elseif :EnzymeForward in broken
            @test_broken Enzyme.gradient(Enzyme.Forward, f, z) ≈ finitediff rtol=rtol atol=atol
            @test Enzyme.gradient(Enzyme.Reverse, f, z) ≈ finitediff rtol=rtol atol=atol
        elseif :Enzyme in broken
            @test_broken Enzyme.gradient(Enzyme.Forward, f, z) ≈ finitediff rtol=rtol atol=atol
            @test_broken Enzyme.gradient(Enzyme.Reverse, f, z) ≈ finitediff rtol=rtol atol=atol
        else
            @test Enzyme.gradient(Enzyme.Forward, f, z) ≈ finitediff rtol=rtol atol=atol
            @test Enzyme.gradient(Enzyme.Reverse, f, z) ≈ finitediff rtol=rtol atol=atol
        end
    return
end
