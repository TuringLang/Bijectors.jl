# For all multivariate distributions, from_vec and to_vec are just the identity function.
from_vec(::D.MultivariateDistribution) = TypedIdentity()
to_vec(::D.MultivariateDistribution) = TypedIdentity()
# which makes vec_length and optic_vec trivial
vec_length(d::D.MultivariateDistribution) = length(d)
# TODO(penelopeysm): We assume here that the axes of the distribution are 1:length(d). This
# is not always true, but we don't (yet) have a good way to determine that... If you're
# reading this, check for updates in:
# https://github.com/JuliaStats/Distributions.jl/issues/734
# https://github.com/JuliaStats/Distributions.jl/pull/2009
function optic_vec(d::D.MultivariateDistribution)
    return [AbstractPPL.@opticof(_[i]) for i in 1:length(d)]
end

# For discrete multivariate distributions, we really can't transform the 'support'.
from_linked_vec(::D.DiscreteMultivariateDistribution) = TypedIdentity()
to_linked_vec(::D.DiscreteMultivariateDistribution) = TypedIdentity()
linked_vec_length(d::D.DiscreteMultivariateDistribution) = vec_length(d)
linked_optic_vec(d::D.DiscreteMultivariateDistribution) = optic_vec(d)
