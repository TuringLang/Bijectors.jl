using Bijectors

using ChainRulesTestUtils
using Combinatorics
using ADTypes
using AbstractPPL: AbstractPPL
using DifferentiationInterface: DifferentiationInterface
using FiniteDifferences
using ForwardDiff
using Functors
using LogExpFunctions

using Random, LinearAlgebra, Test

using Bijectors:
    Shift,
    Scale,
    Logit,
    SimplexBijector,
    PDBijector,
    Permute,
    PlanarLayer,
    RadialLayer,
    Stacked,
    TruncatedBijector

using ChangesOfVariables: ChangesOfVariables
using InverseFunctions: InverseFunctions
using LazyArrays: LazyArrays

const GROUP = get(ENV, "GROUP", "All")

# Non-ForwardDiff backends run under `test/integration_tests/`.

include("test_resources.jl")

# Always include this since it can be useful for other tests.
include("bijectors/utils.jl")

@testset "Bijectors.jl" begin
    if GROUP == "All" || GROUP == "Classic"
        include("interface.jl")
        include("legacy_interface.jl")
        include("normalising_flows.jl")
        include("bijectors/permute.jl")
        include("bijectors/rational_quadratic_spline.jl")
        include("bijectors/named_bijector.jl")
        include("bijectors/leaky_relu.jl")
        include("bijectors/coupling.jl")
        include("bijectors/ordered.jl")
        include("bijectors/pd.jl")
        include("bijectors/reshape.jl")
        include("bijectors/corr.jl")
        include("bijectors/chainrules.jl")
        include("bijectors/product_bijector.jl")
        include("bijectors/named_stacked.jl")
        include("bijectors/stacked.jl")
        include("bijectors/simplex.jl")
        include("bijectors/equality.jl")
        include("bijectors/scale.jl")
        include("bijectors/cdf_quantile.jl")
    end

    if GROUP == "All" || GROUP == "Vector" || GROUP == "VectorProduct"
        # Main-suite non-AD `VectorBijectors.test_all` coverage. `Vector` covers everything
        # except product distributions, `VectorProduct` covers products only, `All` runs both.
        let
            product_only_tags = (
                :products, :nested_product_namedtuple, :type_unstable_products
            )
            selected_tags = if GROUP == "Vector"
                Tuple(t for t in _VECTOR_TAGS if t ∉ product_only_tags)
            elseif GROUP == "VectorProduct"
                product_only_tags
            else
                _VECTOR_TAGS
            end

            @testset "VectorBijectors test_all" begin
                for c in generate_vector_testcases()
                    c.tag in selected_tags || continue
                    run_vector_case(c)
                end
            end

            # Scalar-to-scalar bijectors not exercised by the test_all sweep.
            GROUP != "VectorProduct" && include("vector/cdf_quantile.jl")
        end
    end
end
