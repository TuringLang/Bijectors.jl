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

            test_reverse(find_alpha!, Const, (out, Tout), (xs, Tx), (ys, Ty), (zs, Tz))
        end
    end
end
