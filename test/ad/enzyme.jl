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

            if VERSION >= v"1.11" && Tx <: Const && Ty <: Const && Tz <: Const
                # Rule not picked up by Enzyme on Julia 1.11?!
                # Ref https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2470766968
                @test_throws "LLVM error" test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            else
                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end
        end

        # Batches
        @testset for RT in (Const, BatchDuplicated, BatchDuplicatedNoNeed),
            Tx in (Const, BatchDuplicated),
            Ty in (Const, BatchDuplicated),
            Tz in (Const, BatchDuplicated)

            if VERSION >= v"1.11" && Tx <: Const && Ty <: Const && Tz <: Const
                # Rule not picked up by Enzyme on Julia 1.11?!
                # Ref https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2470766968
                @test_throws "LLVM error" test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            else
                test_forward(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
            end
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

        # Batches
        function find_alpha!(
            out::Vector{<:Real}, x::Vector{<:Real}, y::Vector{<:Real}, z::Vector{<:Real}
        )
            map!(Bijectors.find_alpha, out, x, y, z)
            return nothing
        end
        n = 3
        out = zeros(n)
        xs = randn(n)
        ys = expm1.(randn(n))
        zs = randn(n)
        @testset for Tout in (Const, BatchDuplicated),
            Tx in (Const, BatchDuplicated),
            Ty in (Const, BatchDuplicated),
            Tz in (Const, BatchDuplicated)

            if Tx <: Const && Ty <: Const && Tz <: Const
                test_reverse(find_alpha!, Const, (out, Tout), (xs, Tx), (ys, Ty), (zs, Tz))
            else
                # Not supported by Enzyme: https://github.com/TuringLang/Bijectors.jl/pull/350#issuecomment-2480468728
                @test_throws "LLVM error" test_reverse(find_alpha!, Const, (out, Tout), (xs, Tx), (ys, Ty), (zs, Tz))
            end 
        end
    end
end
