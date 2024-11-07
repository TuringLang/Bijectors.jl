# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")

function test_ad(f, x, broken=(); rtol=1e-6, atol=1e-6)
    for b in broken
        if !(
            b in (
                :ForwardDiff,
                :Zygote,
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

    if (AD == "All" || AD == "Enzyme")
        forward_broken = :EnzymeForward in broken || :Enzyme in broken
        reverse_broken = :EnzymeReverse in broken || :Enzyme in broken
        if !(:EnzymeForwardCrash in broken)
            if forward_broken
                @test_broken(
                    collect(et, Enzyme.gradient(Enzyme.Forward, f, x)[1]) ≈ finitediff,
                    rtol = rtol,
                    atol = atol
                )
            else
                @test(
                    collect(et, Enzyme.gradient(Enzyme.Forward, f, x)[1]) ≈ finitediff,
                    rtol = rtol,
                    atol = atol
                )
            end
        end

        if !(:EnzymeReverseCrash in broken)
            if reverse_broken
                @test_broken(
                    collect(et, Enzyme.gradient(Enzyme.Reverse, f, x)[1]) ≈ finitediff,
                    rtol = rtol,
                    atol = atol
                )
            else
                @test(
                    collect(et, Enzyme.gradient(Enzyme.Reverse, f, x)[1]) ≈ finitediff,
                    rtol = rtol,
                    atol = atol
                )
            end
        end
    end

    if (AD == "All" || AD == "Mooncake") && VERSION >= v"1.10"
        try
            Mooncake.build_rrule(f, x)
        catch exc
            # TODO(penelopeysm):
            # @test_throws AssertionError (expr...) doesn't work, unclear why
            # We use `isdefined` here since `hasproperty` for modules is not consistent with `getproperty`
            # Ref https://github.com/JuliaLang/julia/issues/47150
            if isdefined(Mooncake, :MooncakeRuleCompilationError)
                @test exc isa getproperty(Mooncake, :MooncakeRuleCompilationError)
            else
                @test exc isa AssertionError
            end
        end
        # TODO: The above @test_throws happens because of 
        # https://github.com/compintell/Mooncake.jl/issues/319. If that test
        # fails, it probably means that the issue was fixed, in which case
        # we can remove that block and uncomment the following instead.

        # rule = Mooncake.build_rrule(f, x)
        # if :Mooncake in broken
        #     @test_broken (
        #         isapprox(
        #             Mooncake.value_and_gradient!!(rule, f, x)[2][2],
        #             finitediff;
        #             rtol=rtol,
        #             atol=atol,
        #         )
        #     )
        # else
        #     @test(
        #         isapprox(
        #             Mooncake.value_and_gradient!!(rule, f, x)[2][2],
        #             finitediff;
        #             rtol=rtol,
        #             atol=atol,
        #         )
        #     )
        # end
    end

    return nothing
end
