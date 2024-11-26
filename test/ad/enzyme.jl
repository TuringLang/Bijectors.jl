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

            # Rule not picked up by Enzyme on Julia 1.11?!
            # Ref https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2470766968
            if VERSION >= v"1.11" && Tx <: Const && Ty <: Const && Tz <: Const
                continue
            end

            test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
        end

        # Batches
        @testset for RT in (Const, BatchDuplicated, BatchDuplicatedNoNeed),
            Tx in (Const, BatchDuplicated),
            Ty in (Const, BatchDuplicated),
            Tz in (Const, BatchDuplicated)

            # Rule not picked up by Enzyme on Julia 1.11?!
            # Ref https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2470766968
            if VERSION >= v"1.11" && Tx <: Const && Ty <: Const && Tz <: Const
                continue
            end

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
