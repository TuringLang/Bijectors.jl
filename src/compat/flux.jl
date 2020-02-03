import Flux
Flux.trainable(bn::InvertibleBatchNorm) = (bn.b, bn.logs)
Flux.@functor InvertibleBatchNorm
