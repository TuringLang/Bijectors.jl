# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")

function test_ad(f, x, broken=(); rtol=1e-6, atol=1e-6)
    for b in broken
        if !(b in (:ForwardDiff, :Zygote, :Tapir, :ReverseDiff, :Enzyme, :EnzymeForward, :EnzymeReverse))
            error("Unknown broken AD backend: $b")
        end
    end

    finitediff = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]
    et = eltype(finitediff)

    if AD == "All" || AD == "ForwardDiff"
        if :ForwardDiff in broken
            @test_broken ForwardDiff.gradient(f, x) ≈ finitediff rtol = rtol atol = atol
        else
            @test ForwardDiff.gradient(f, x) ≈ finitediff rtol = rtol atol = atol
        end
    end

    if AD == "All" || AD == "Zygote"
        if :Zygote in broken
            @test_broken Zygote.gradient(f, x)[1] ≈ finitediff rtol = rtol atol = atol
        else
            ∇zygote = Zygote.gradient(f, x)[1]
            @test (all(finitediff .== 0) && ∇zygote === nothing) ||
                  isapprox(∇zygote, finitediff; rtol=rtol, atol=atol)
        end
    end

    if AD == "All" || AD == "ReverseDiff"
        if :ReverseDiff in broken
            @test_broken ReverseDiff.gradient(f, x) ≈ finitediff rtol = rtol atol = atol
        else
            @test ReverseDiff.gradient(f, x) ≈ finitediff rtol = rtol atol = atol
        end
    end

    if AD == "All" || AD == "Enzyme"
        if :EnzymeReverse in broken
            @test(
                collect(et, Enzyme.gradient(Enzyme.Forward, f, x)) ≈ finitediff,
                rtol = rtol,
                atol = atol
            )
            @test_broken(
                Enzyme.gradient(Enzyme.Reverse, f, x) ≈ finitediff, rtol = rtol, atol = atol
            )
        elseif :EnzymeForward in broken
            @test_broken(
                collect(et, Enzyme.gradient(Enzyme.Forward, f, x)) ≈ finitediff,
                rtol = rtol,
                atol = atol
            )
            @test(
                Enzyme.gradient(Enzyme.Reverse, f, x) ≈ finitediff, rtol = rtol, atol = atol
            )
        elseif :Enzyme in broken
            @test_broken(
                collect(et, Enzyme.gradient(Enzyme.Forward, f, x)) ≈ finitediff,
                rtol = rtol,
                atol = atol
            )
            @test_broken(
                Enzyme.gradient(Enzyme.Reverse, f, x) ≈ finitediff, rtol = rtol, atol = atol
            )
        else
            @test(
                collect(et, Enzyme.gradient(Enzyme.Forward, f, x)) ≈ finitediff,
                rtol = rtol,
                atol = atol
            )
            @test(
                Enzyme.gradient(Enzyme.Reverse, f, x) ≈ finitediff, rtol = rtol, atol = atol
            )
        end
    end

    if (AD == "All" || AD == "Tapir") && VERSION >= v"1.10"
        rule = Tapir.build_rrule(f, x; safety_on=false)
        if :tapir in broken
            @test_broken(
                isapprox(
                    Tapir.value_and_gradient!!(rule, f, x)[2][2],
                    finitediff;
                    rtol=rtol,
                    atol=atol,
                )
            )
        else
            @test(
                isapprox(
                    Tapir.value_and_gradient!!(rule, f, x)[2][2],
                    finitediff;
                    rtol=rtol,
                    atol=atol,
                )
            )
        end
    end

    return nothing
end
