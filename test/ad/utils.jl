# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")

function test_ad(f, x, broken=(); rtol=1e-6, atol=1e-6)
    for b in broken
        if !(
            b in (
                :ForwardDiff,
                :Mooncake,
                :ReverseDiff,
                :Enzyme,
                :EnzymeForward,
                :EnzymeReverse,
                # The `Crash` ones indicate that the error will cause a Julia crash, and
                # thus we can't even run `@test_broken on it.
                :EnzymeForwardCrash,
                :EnzymeReverseCrash,
            )
        )
            error("Unknown broken AD backend: $b")
        end
    end

    finitediff = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]

    if AD == "All" || AD == "ForwardDiff"
        if :ForwardDiff in broken
            @test_broken ForwardDiff.gradient(f, x) ≈ finitediff rtol = rtol atol = atol
        else
            @test ForwardDiff.gradient(f, x) ≈ finitediff rtol = rtol atol = atol
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
        forward_broken = :EnzymeForward in broken || :Enzyme in broken
        reverse_broken = :EnzymeReverse in broken || :Enzyme in broken
        if !(:EnzymeForwardCrash in broken)
            if forward_broken
                @test_broken(
                    Enzyme.gradient(Forward, Enzyme.Const(f), x)[1] ≈ finitediff,
                    rtol = rtol,
                    atol = atol
                )
            else
                @test(
                    Enzyme.gradient(Forward, Enzyme.Const(f), x)[1] ≈ finitediff,
                    rtol = rtol,
                    atol = atol
                )
            end
        end

        if !(:EnzymeReverseCrash in broken)
            if reverse_broken
                @test_broken(
                    Enzyme.gradient(set_runtime_activity(Reverse), Enzyme.Const(f), x)[1] ≈
                        finitediff,
                    rtol = rtol,
                    atol = atol
                )
            else
                @test(
                    Enzyme.gradient(set_runtime_activity(Reverse), Enzyme.Const(f), x)[1] ≈
                        finitediff,
                    rtol = rtol,
                    atol = atol
                )
            end
        end
    end

    if AD == "All" || AD == "Mooncake"
        rule = Mooncake.build_rrule(f, x)
        if :Mooncake in broken
            @test_broken isapprox(
                Mooncake.value_and_gradient!!(rule, f, x)[2][2],
                finitediff;
                rtol=rtol,
                atol=atol,
            )
        else
            @test isapprox(
                Mooncake.value_and_gradient!!(rule, f, x)[2][2],
                finitediff;
                rtol=rtol,
                atol=atol,
            )
        end
    end

    return nothing
end
