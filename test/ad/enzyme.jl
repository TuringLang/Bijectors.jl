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
        @testset for RT in (Const, Active),
            Tx in (Const, Active),
            Ty in (Const, Active),
            Tz in (Const, Active)

            test_reverse(Bijectors.find_alpha, RT, (x, Tx), (y, Ty), (z, Tz))
        end
    end
end
