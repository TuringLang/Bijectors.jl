module VectorBijectors

using ..Bijectors: Bijectors as B
using Distributions: Distributions
const D = Distributions
import ChangesOfVariables: with_logabsdet_jacobian
import InverseFunctions: inverse

include("interface.jl")

include("univariate/univariate.jl")
include("univariate/identities.jl")
include("univariate/positive.jl")
include("univariate/truncated.jl")

include("multivariate/multivariate.jl")
include("multivariate/identities.jl")
include("multivariate/simplex.jl")

# Public interface
export from_vec
export to_vec
export from_linked_vec
export to_linked_vec
export vec_length
export linked_vec_length
# re-exports
export with_logabsdet_jacobian
export inverse

end # module VectorBijectors
