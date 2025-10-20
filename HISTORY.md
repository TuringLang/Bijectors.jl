# 0.15.11

Bijectors for ProductNamedTupleDistribution are now implemented.

`Bijectors.output_size` is now exported. This function provides information about the size of transformed variables. There are two main invocations:

  - `output_size(b, input_size::Tuple)` returns the size of the output of `b`, given an input that has size `input_size`.
  - `output_size(b, dist::Distribution)` returns the size of the output of `b`, given an input sampled from distribution `dist`. For most distributions this is implemented by calling `output_size(b, size(dist))`; however, ProductNamedTupleDistribution does not implement `size`, so this method is necessary.
