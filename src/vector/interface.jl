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
