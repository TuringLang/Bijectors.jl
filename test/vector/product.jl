module VBProductTests

using Distributions
using LinearAlgebra
using FillArrays: Fill
using Test
using Bijectors.VectorBijectors
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake
using Enzyme: Enzyme, set_runtime_activity, Const, Forward, Reverse

adtypes = [
    DI.AutoReverseDiff(),
    DI.AutoReverseDiff(; compile=true),
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    # Need runtime activity for some reason.
    # TODO(penelopeysm): Report upstream
    DI.AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    DI.AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
]

# Enzyme segfaults on 1.12 + Windows.
# https://github.com/EnzymeAD/Enzyme.jl/issues/2986
if VERSION >= v"1.12-" && Sys.iswindows()
    filter!(a -> !(a isa DI.AutoEnzyme), adtypes)
end

# These are purposely chosen because the vec_length output is the same but
# linked_vec_length differs.
m2 = MvNormal(zeros(2), I)
d2 = Dirichlet(ones(2))

p1t = product_distribution(Normal(), Beta(2, 2))
p2t = product_distribution(m2, d2)
p1a = product_distribution(fill(Beta(2, 2), 2))
p2a = product_distribution(fill(d2, 2))

products = [
    # Tuples
    product_distribution(Normal()),
    product_distribution(Normal(), Normal()),
    product_distribution(Normal(), Beta(2, 2)),
    product_distribution(Beta(2, 2), Exponential()),
    product_distribution(m2, d2),
    product_distribution(m2, d2, m2, d2),
    # Vectors of univariate (Distributions.Product)
    product_distribution(fill(Normal(), 2)), # This is actually an MvNormal in disguise
    product_distribution(fill(Beta(2, 2), 2)),
    product_distribution([Uniform(0, 1), Uniform(1, 2), Uniform(2, 3)]),
    product_distribution(Fill(Uniform(1, 2), 2)),
    # >1D arrays, or vectors of >1D distributions (Distributions.ProductDistribution)
    product_distribution(fill(Normal(), 2, 2)),
    product_distribution(Fill(Uniform(1, 2), 2, 2)),
    product_distribution(fill(m2, 2, 2)),
    product_distribution(Fill(m2, 2, 2)),
    product_distribution(fill(d2, 2, 2)),
    # NamedTuples
    product_distribution((a=Normal(), b=Beta(2, 2))),
    product_distribution((a=Normal(), b=Dirichlet(ones(2)))),
    product_distribution((a=Normal(), b=product_distribution(fill(Beta(2, 2), 2)))),
    # Nested
    product_distribution(fill(p1t, 2)),
    product_distribution(fill(p1t, 2, 2)),
    product_distribution(p2t, p2t, p2t),
    product_distribution(fill(p2t, 2)),
    product_distribution(fill(p2t, 2, 2)),
    product_distribution(fill(p1a, 2)),
    product_distribution(fill(p1a, 2, 2)),
    product_distribution(p2a, p2a, p2a),
    product_distribution(fill(p2a, 2)),
    product_distribution(fill(p2a, 2, 2)),
]

nested_product_namedtuple = [
    # This one is very weird: on Julia 1.10 (and only on Julia 1.10), @inferred to_vec(d)
    # fails, but @code_warntype to_vec(d) is completely type stable. Almost certainly a
    # Julia bug or at least a weird quirk
    product_distribution((a=Normal(), b=product_distribution((c=Normal(), d=Beta(2, 2))))),
]

heterogeneous_products = [
    # These contain heterogeneous arrays, which means that the construction of the bijector
    # is type unstable. I don't think it's possible to fix this, but someone should probably at
    # least try.
    product_distribution([Normal(), Beta(2, 2), Exponential()]),
    product_distribution([Normal() Beta(2, 2); Exponential() Uniform(-1, 1)]),
    product_distribution([m2 d2; m2 d2]),
]

enzyme_failures = [
    # These work generally but fail with Enzyme -- should probably be reported upstream
    product_distribution(p1t, p1t, p1t),
    product_distribution(p1a, p1a, p1a),
]

@testset "Product distributions" begin
    for d in products
        VectorBijectors.test_all(d; adtypes=adtypes, expected_zero_allocs=())
    end

    for d in nested_product_namedtuple
        VectorBijectors.test_all(
            d;
            adtypes=adtypes,
            expected_zero_allocs=(),
            test_construction_type_stable=(VERSION >= v"1.11-"),
        )
    end

    for d in heterogeneous_products
        VectorBijectors.test_all(
            d; adtypes=adtypes, expected_zero_allocs=(), test_construction_type_stable=false
        )
    end

    no_enzyme_adtypes = filter(adtype -> !(adtype isa DI.AutoEnzyme), adtypes)
    for d in enzyme_failures
        VectorBijectors.test_all(
            d;
            adtypes=no_enzyme_adtypes,
            expected_zero_allocs=(),
            test_construction_type_stable=false,
        )
    end
end

end # module VBProductTests
