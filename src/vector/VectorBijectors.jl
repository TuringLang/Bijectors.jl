module VectorBijectors

using AbstractPPL: AbstractPPL
using ..Bijectors: Bijectors as B
using Distributions: Distributions
const D = Distributions
import ChangesOfVariables: with_logabsdet_jacobian
import InverseFunctions: inverse

include("interface.jl")
export from_vec
export to_vec
export from_linked_vec
export to_linked_vec
export vec_length
export linked_vec_length
export optic_vec
export linked_optic_vec
# re-exports
export with_logabsdet_jacobian
export inverse

include("common.jl")

include("univariate/univariate.jl")
include("univariate/identities.jl")
include("univariate/positive.jl")
include("univariate/truncated.jl")

include("multivariate/multivariate.jl")
include("multivariate/identities.jl")
include("multivariate/mvlognormal.jl")
include("multivariate/simplex.jl")

include("matrix/matrix.jl")
include("matrix/normal.jl")
include("matrix/posdef.jl")
include("matrix/lkj.jl")

include("order/order.jl")
include("reshaped/reshaped.jl")
include("cholesky/cholesky.jl")

include("product/product.jl")

# Put last to avoid cluttering namespace
include("test_utils.jl")

end # module VectorBijectors
