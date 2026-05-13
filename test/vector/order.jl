@testset "Order statistics" begin
    for c in generate_testcases(Val(:order_orderstatistic))
        run_vector_case(c, adtypes)
    end
    # ReverseDiff can't differentiate JointOrderStatistics because of the heavy setindex!
    # usage. https://github.com/JuliaDiff/ReverseDiff.jl/issues/43
    joint_adtypes = [AutoMooncake(), AutoMooncakeForward()]
    for c in generate_testcases(Val(:order_joint))
        run_vector_case(c, joint_adtypes)
    end
    for c in generate_testcases(Val(:order_ordered))
        run_vector_case(c, adtypes)
    end
end
