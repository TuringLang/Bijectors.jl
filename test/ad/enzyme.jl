module BijectorsEnzymeRulesTests

using Bijectors
using Enzyme
using EnzymeTestUtils: test_forward, test_reverse
using Test

# This entire test suite is broken on 1.11.
#
# https://github.com/EnzymeAD/Enzyme.jl/issues/2121
# https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2470766968
#
# Ideally we'd use `@test_throws`. However, that doesn't work because
# `test_forward` itself calls `@test`, and the error is captured by that
# `@test`, not our `@test_throws`. Consequently `@test_throws` doesn't actually
# see any error. Weird Julia behaviour.

@static if VERSION < v"1.11"
    @testset "Enzyme: Bijectors.find_alpha" begin
        x = randn()
        y = expm1(randn())
        z = randn()

        @testset "forward" begin
            # No batches
            @testset for RT in (Const, Duplicated, DuplicatedNoNeed),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated),
                Tz in (Const, Duplicated)

                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end

            # Batches
            @testset for RT in (Const, BatchDuplicated, BatchDuplicatedNoNeed),
                Tx in (Const, BatchDuplicated),
                Ty in (Const, BatchDuplicated),
                Tz in (Const, BatchDuplicated)

                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end
        end
        @testset "reverse" begin
            # No batches
            @testset for RT in (Const, Active),
                Tx in (Const, Active),
                Ty in (Const, Active),
                Tz in (Const, Active)

                test_reverse(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end

            # TODO: Test batch mode
            # This is a bit problematic since Enzyme does not support all combinations of activities currently
            # https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2480468728
        end
    end
end

end
