# 0.15.15

Fixed a case where calling `inverse(VecCholeskyBijector())` on a wrong input would cause a stack overflow instead of a more normal error.

# 0.15.14

Added a new documentation page describing how to implement custom bijectors (this release contains no code changes; it only exists to make sure that the 'stable' docs are updated).

# 0.15.13

Exports extra functionality that should probably have been exported, namely `ordered`, `isinvertible`, and `columnwise`, from Bijectors.jl.

The docs have been thoroughly restructured.

# 0.15.12

Improved implementation of the Enzyme rule for `Bijectors.find_alpha`.

# 0.15.11

Bijectors for ProductNamedTupleDistribution are now implemented.

`Bijectors.output_size` is now exported. This function provides information about the size of transformed variables. There are two main invocations:

  - `output_size(b, input_size::Tuple)` returns the size of the output of `b`, given an input that has size `input_size`.
  - `output_size(b, dist::Distribution)` returns the size of the output of `b`, given an input sampled from distribution `dist`. For most distributions this is implemented by calling `output_size(b, size(dist))`; however, ProductNamedTupleDistribution does not implement `size`, so this method is necessary.
