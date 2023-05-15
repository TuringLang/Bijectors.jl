struct Reshape{S1,S2} <: Bijector
    in_shape::S1
    out_shape::S2
end

inverse(b::Reshape) = Reshape(b.out_shape, b.in_shape)

logabsdetajc(::Reshape, x) = zero(eltype(x))
transform(b::Reshape, x) = reshape(x, b.out_shape)
