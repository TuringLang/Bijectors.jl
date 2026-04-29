"""
    VectorBijectors.from_vec(d::Distribution)

Returns a function that can be used to convert a vectorised sample from `d` back to its
original form.

`from_vec(d)` is the inverse of `to_vec(d)`.

## Examples

```jldoctest
julia> using Bijectors.VectorBijectors: from_vec; using Distributions

julia> d = Beta(2, 2); from_vec(d)([0.5])
0.5

ulia> d = product_distribution((a = Normal(), b = Beta(2, 2))); from_vec(d)([0.2, 0.5])
(a = 0.2, b = 0.5)
```
"""
function from_vec end

"""
    VectorBijectors.to_vec(d::Distribution)

Returns a function that can be used to vectorise a sample from `d`.

`to_vec(d)` is the inverse of `from_vec(d)`.

## Examples

```jldoctest
julia> using Bijectors.VectorBijectors: to_vec; using Distributions

julia> d = Beta(2, 2); to_vec(d)(0.5)
1-element Vector{Float64}:
 0.5

julia> d = product_distribution((a = Normal(), b = Beta(2, 2))); to_vec(d)((a = 0.2, b = 0.5))
2-element Vector{Float64}:
 0.2
 0.5
```
"""
function to_vec end

"""
    VectorBijectors.optic_vec(d::Distribution)

Returns a vector of optics (from AbstractPPL.jl), which describe how each element in the
vectorised sample from `d` can be accessed from the original sample.

For example, if `d = MvNormal(zeros(3), I)`, then `optic_vec(d)` would return a vector of
`[@opticof(_[1]), @opticof(_[2]), @opticof(_[3])]`. The length of this vector would be equal
to `vec_length(d)`.

If `optics = optic_vec(d)`, then for any sample `x ~ d` and its vectorised form `v =
to_vec(d)(x)`, it should hold that `v[i] == optics[i](x)` for all `i`.

## Examples

For the `Beta` case, the return value of an empty optic indicates that 'the first element of
the vector is the sample itself'.

For the `product_distribution` case, the return value of `Optic(.a)` indicates that 'the
first element of the vector is the `.a` component of the sample', and similarly for
`Optic(.b)`.

```jldoctest
julia> using Bijectors.VectorBijectors: optic_vec; using Distributions

julia> d = Beta(2, 2); optic_vec(d)
1-element Vector{AbstractPPL.Iden}:
 Optic()

julia> d = product_distribution((a = Normal(), b = Beta(2, 2))); optic_vec(d)
2-element Vector{AbstractPPL.Property{_A, AbstractPPL.Iden} where _A}:
 Optic(.a)
 Optic(.b)
```
"""
function optic_vec end

"""
    VectorBijectors.from_linked_vec(d::Distribution)

Returns a function that can be used to convert an unconstrained vector back to a sample from
`d`.

`from_linked_vec(d)` is the inverse of `to_linked_vec(d)`.

## Examples

```jldoctest
julia> using Bijectors.VectorBijectors: from_linked_vec; using Distributions

julia> d = Beta(2, 2); from_linked_vec(d)([1.0])
0.7310585786300049

julia> d = product_distribution((a = Normal(), b = Beta(2, 2))); from_linked_vec(d)([0.2, 1.0])
(a = 0.2, b = 0.7310585786300049)
```
"""
function from_linked_vec end

"""
    VectorBijectors.to_linked_vec(d::Distribution)

Returns a function that can be used to convert a sample from `d` to an unconstrained vector.

`to_linked_vec(d)` is the inverse of `from_linked_vec(d)`.

## Examples

```jldoctest
julia> using Bijectors.VectorBijectors: to_linked_vec; using Distributions

julia> d = Beta(2, 2); to_linked_vec(d)(0.5)
1-element Vector{Float64}:
 0.0

julia> d = product_distribution((a = Normal(), b = Beta(2, 2))); to_linked_vec(d)((a = 0.2, b = 0.5))
2-element Vector{Float64}:
 0.2
 0.0
```
"""
function to_linked_vec end

"""
    VectorBijectors.linked_optic_vec(d::Distribution)

Returns a vector of optics (from AbstractPPL.jl), which describe how each element in the
unconstrained vector representation of a sample from `d` is related to the original sample.

This is not always well-defined. For example, consider a `Dirichlet` distribution, with
three components. The unconstrained vector representation would have two elements. However,
these two elements do not correspond to specific components of the original sample, since
all three components are interdependent (they must sum to one). In such cases, this function
should return a vector of two `nothing`s.

However, for a distribution like `MvNormal`, this function would return a vector of
`[@opticof(_[1]), @opticof(_[2]), @opticof(_[3])]`, similar to `optic_vec`. That is because
the first element of the unconstrained vector is solely determined by the first component of
the original sample, the second element by the second component, and so on.

Note that, unlike `optic_vec`, the first element of the linked vector does not necessarily
have to be _exactly equal_ to the first component of the original sample, as there may have
been a transformation applied. It merely needs to be _determined by_ the first component
(and _only_ the first component).

## Examples

In this case since linking does not affect the 'provenance' of the sample (i.e., for the
product distribution the first element of the linked vector is still determined by `.a` and
the second by `.b`), the return value of `linked_optic_vec` is the same as that of
`optic_vec`.

```jldoctest
julia> using Bijectors.VectorBijectors: linked_optic_vec; using Distributions

julia> d = Beta(2, 2); linked_optic_vec(d)
1-element Vector{AbstractPPL.Iden}:
 Optic()

julia> d = product_distribution((a = Normal(), b = Beta(2, 2))); linked_optic_vec(d)
2-element Vector{AbstractPPL.Property{_A, AbstractPPL.Iden} where _A}:
 Optic(.a)
 Optic(.b)
```
"""
function linked_optic_vec end

"""
    VectorBijectors.vec_length(d::Distribution)

Returns the length of the vector representation of a sample from `d`, i.e.,
`length(to_vec(d)(rand(d)))`. However, it does this without actually drawing a sample.

## Examples

```jldoctest
julia> using Bijectors.VectorBijectors: vec_length; using Distributions

julia> d = Beta(2, 2); vec_length(d)
1

julia> d = product_distribution((a = Normal(), b = Beta(2, 2))); vec_length(d)
2
```
"""
function vec_length end

"""
    VectorBijectors.linked_vec_length(d::Distribution)

Returns the length of the unconstrained vector representation of a sample from `d`, i.e.,
`length(to_linked_vec(d)(rand(d)))`. However, it does this without actually drawing a
sample.

## Examples

```jldoctest
julia> using Bijectors.VectorBijectors: linked_vec_length; using Distributions

julia> d = Beta(2, 2); linked_vec_length(d)
1

julia> d = product_distribution((a = Normal(), b = Beta(2, 2))); linked_vec_length(d)
2
```
"""
function linked_vec_length end

for f in (:from_vec, :to_vec, :from_linked_vec, :to_linked_vec)
    @eval begin
        function B.with_logabsdet_jacobian(::typeof(VectorBijectors.$f), ::Any)
            return error(
                "`" *
                string($f) *
                "` is not a transform itself. Perhaps you meant to call `with_logabsdet_jacobian(" *
                string($f) *
                "(dist), x)`?",
            )
        end
    end
end
