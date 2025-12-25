"""
    VectorBijectors.from_vec(d::Distribution)

Returns a function that can be used to convert a vectorised sample from `d` back to its
original form.
"""
function from_vec end

"""
    VectorBijectors.to_vec(d::Distribution)

Returns a function that can be used to vectorise a sample from `d`.
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
"""
function optic_vec end

"""
    VectorBijectors.from_linked_vec(d::Distribution)

Returns a function that can be used to convert an unconstrained vector back to a sample from
`d`.
"""
function from_linked_vec end

"""
    VectorBijectors.to_linked_vec(d::Distribution)

Returns a function that can be used to convert a sample from `d` to an unconstrained vector.
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
"""
function linked_optic_vec end

"""
    VectorBijectors.vec_length(d::Distribution)

Returns the length of the vector representation of a sample from `d`, i.e.,
`length(to_vec(d)(rand(d)))`. However, it does this without actually drawing a sample.
"""
function vec_length end

"""
    VectorBijectors.linked_vec_length(d::Distribution)

Returns the length of the unconstrained vector representation of a sample from `d`, i.e.,
`length(to_linked_vec(d)(rand(d)))`. However, it does this without actually drawing a
sample.
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
