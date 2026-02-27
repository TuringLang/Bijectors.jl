# Product distributions of tuples, arrays, and NamedTuples.
#
# There is a LOT of code repetition here! Unfortunately that's because most of these
# functions have to be generated functions in order to ensure type stability (particularly
# for NamedTuples; for the Tuple case, we can often get away with a non-generated function
# since Julia performs union splitting, but that causes issues with Enzyme; so we just make
# everything a generated function).

struct ProductVecTransform{TTrf,Trng,D}
    "A collection of vectorisation transforms, one for each component of the product
    distribution. These may either be `to_vec` or `to_linked_vec` transforms, which in turn
    determines the overall behaviour of this transform.

    The collection type (e.g. Tuple, Array, or NamedTuple) reflects the underlying structure
    of the product distribution. It should always be the case that the collection type of
    `transforms` is the same as the collection type of `ranges`."
    transforms::TTrf
    "A collection of ranges which specify the output range for each component of the product
    distribution's bijectors."
    ranges::Trng
    "Size of each distribution inside the product distribution. Distributions.jl enforces
    that all distributions in a product distribution have the same size."
    base_size::D
end
struct ProductVecInvTransform{TTrf,Trng,D}
    "A collection of inverse vectorisation transforms, one for each component of the product
    distribution. These may either be `from_vec` or `from_linked_vec` transforms, which in
    turn determines the overall behaviour of this transform."
    transforms::TTrf
    "A collection of ranges which specify the input range for each component of the product
    distribution's bijectors."
    ranges::Trng
    "Size of each distribution inside the product distribution. Distributions.jl enforces
    that all distributions in a product distribution have the same size."
    base_size::D
end
function inverse(t::ProductVecTransform)
    return ProductVecInvTransform(inverse.(t.transforms), t.ranges, t.base_size)
end
function inverse(t::ProductVecInvTransform)
    return ProductVecTransform(inverse.(t.transforms), t.ranges, t.base_size)
end

# zero(T) but with fallback for non-numeric T.
_fzero(::Type{T}) where {T<:Number} = zero(T)
_fzero(@nospecialize(T)) = 0.0

"""
Return an object that can be iterated over to obtain the values for each distribution
in the product distribution.
"""
function _get_val_iterator(::ProductVecTransform{<:Any,<:Any,Tuple{}}, x::AbstractArray)
    # if the base_size is an empty tuple, then it's a univariate distribution,
    # and the values are just the elements of `x`.
    return x
end
function _get_val_iterator(
    ::ProductVecTransform{<:Any,<:Any,NTuple{N,Int}}, x::AbstractArray{T,MplusN}
) where {T,MplusN,N}
    # Multivariate case. The distribution itself has dimension N, and has been expanded
    # by M extra dimensions (e.g. fill(MvNormal(...), 3, 3, 3) would have M=3 and N=2).
    # In the sample, the N dimensions come first, followed by the M dimensions.
    M = MplusN - N
    dims = ntuple(i -> i + N, M)
    return eachslice(x; dims=dims)
end

@generated function with_logabsdet_jacobian(
    t::ProductVecTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}},
    x::AbstractArray{T},
) where {P,N,T}
    # P = number of distributions in the product distribution
    # N = dimension of each distribution
    exprs = []
    push!(exprs, :(total_length = sum(length, t.ranges)))
    push!(exprs, :(logjac = _fzero(T)))
    push!(exprs, :(y = Vector{T}(undef, total_length)))
    colons = fill(:, N)
    y_syms = Symbol.(:y, 1:P)
    logjac_syms = Symbol.(:lj, 1:P)
    for (i, (y_sym, lj_sym)) in enumerate(zip(y_syms, logjac_syms))
        if N == 0
            push!(
                exprs,
                :(($y_sym, $lj_sym) = with_logabsdet_jacobian(t.transforms[$i], x[$i])),
            )
        else
            push!(
                exprs,
                :(
                    ($y_sym, $lj_sym) = with_logabsdet_jacobian(
                        t.transforms[$i], view(x, $colons..., $i)
                    )
                ),
            )
        end
        push!(exprs, :(y[t.ranges[$i]] .= $y_sym))
        push!(exprs, :(logjac += $lj_sym))
    end
    push!(exprs, :(return (y, logjac)))
    return Expr(:block, exprs...)
end

# Regarding formatting: the autoformatter will turn this into
#     @generated function (
#         t::...
#     )(x::...)
# which is broken on 1.12 because of a parser bug.
# https://github.com/JuliaLang/JuliaSyntax.jl/pull/580
#! format: off
@generated function (t::ProductVecTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}})(
    x::AbstractArray{T}
) where {P,N,T}
#! format: on
    exprs = []
    push!(exprs, :(total_length = sum(length, t.ranges)))
    push!(exprs, :(y = Vector{T}(undef, total_length)))
    colons = fill(:, N)
    for i in 1:P
        if N == 0
            push!(exprs, :(y[t.ranges[$i]] = t.transforms[$i](x[$i])))
        else
            push!(exprs, :(y[t.ranges[$i]] .= t.transforms[$i](view(x, $colons..., $i))))
        end
    end
    push!(exprs, :(return y))
    return Expr(:block, exprs...)
end

function with_logabsdet_jacobian(
    t::ProductVecTransform{<:AbstractArray}, x::AbstractArray{T}
) where {T}
    total_length = sum(length, t.ranges)
    logjac = _fzero(T)
    y = Vector{T}(undef, total_length)
    val_iterator = _get_val_iterator(t, x)
    for (trf, r, val) in zip(t.transforms, t.ranges, val_iterator)
        # TODO(penelopeysm): The `xvec[r] .= xr` is inefficient. We could do better by having
        # mutating versions of bijectors.
        xr, lj = with_logabsdet_jacobian(trf, val)
        y[r] .= xr
        logjac += lj
    end
    return y, logjac
end
function (t::ProductVecTransform{<:AbstractArray})(x::AbstractArray{T}) where {T}
    total_length = sum(length, t.ranges)
    y = Vector{T}(undef, total_length)
    val_iterator = _get_val_iterator(t, x)
    for (trf, r, val) in zip(t.transforms, t.ranges, val_iterator)
        y[r] .= trf(val)
    end
    return y
end

@generated function with_logabsdet_jacobian(
    t::ProductVecTransform{<:NamedTuple{names}}, x::NamedTuple{names}
) where {names}
    vcat_args = [Symbol("y_$nm") for nm in names]
    lj_args = [Symbol("lj_$nm") for nm in names]
    exprs = []
    for (nm, y_nm, lj_nm) in zip(names, vcat_args, lj_args)
        push!(exprs, :(($y_nm, $lj_nm) = with_logabsdet_jacobian(t.transforms.$nm, x.$nm)))
    end
    push!(exprs, :(return (vcat($(vcat_args...)), +($(lj_args...)))))
    return Expr(:block, exprs...)
end
# See above for note about formatting
#! format: off
@generated function (t::ProductVecTransform{<:NamedTuple{names}})(
    x::NamedTuple{names}
) where {names}
#! format: on
    expr = Expr(:tuple)
    for nm in names
        push!(expr.args, :(t.transforms.$nm(x.$nm)))
    end
    return :(vcat($expr...))
end

@generated function _set_lastindex!(
    x::AbstractArray{T,N}, i::CartesianIndex{M}, val
) where {T,M,N}
    colons = fill(:, N - M)
    return quote
        x[$colons..., i.I...] = val
    end
end

# Generalisation of CartesianIndices to include tuples.
_cartesian_indices(::NTuple{N,Any}) where {N} = CartesianIndices((N,))
_cartesian_indices(x::AbstractArray) = CartesianIndices(x)

@generated function with_logabsdet_jacobian(
    t::ProductVecInvTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}},
    y::AbstractVector{T},
) where {P,N,T}
    # P = number of distributions in the product distribution
    # N = dimension of each distribution
    exprs = []
    push!(exprs, :(x = Array{T}(undef, t.base_size..., P)))
    push!(exprs, :(logjac = _fzero(T)))
    colons = fill(:, N)
    x_syms = Symbol.(:x, 1:P)
    lj_syms = Symbol.(:lj, 1:P)
    for (i, (x_sym, lj_sym)) in enumerate(zip(x_syms, lj_syms))
        push!(
            exprs,
            :(
                ($x_sym, $lj_sym) = with_logabsdet_jacobian(
                    t.transforms[$i], view(y, t.ranges[$i])
                )
            ),
        )
        push!(exprs, :(x[$colons..., $i] = $x_sym))
        push!(exprs, :(logjac += $lj_sym))
    end
    push!(exprs, :(return (x, logjac)))
    return Expr(:block, exprs...)
end

# See above for note about formatting
#! format: off
@generated function (t::ProductVecInvTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}})(
    y::AbstractVector{T}
) where {P,N,T}
#! format: on
    # P = number of distributions in the product distribution
    # N = dimension of each distribution
    exprs = []
    push!(exprs, :(x = Array{T}(undef, t.base_size..., P)))
    colons = fill(:, N)
    for i in 1:P
        push!(exprs, :(x[$colons..., $i] = t.transforms[$i](view(y, t.ranges[$i]))))
    end
    push!(exprs, :(return x))
    return Expr(:block, exprs...)
end

function with_logabsdet_jacobian(
    t::ProductVecInvTransform{<:AbstractArray}, y::AbstractVector{T}
) where {T}
    x = Array{T}(undef, t.base_size..., size(t.transforms)...)
    logjac = _fzero(T)
    idxs = _cartesian_indices(t.transforms)
    for (idx, trf, r) in zip(idxs, t.transforms, t.ranges)
        xr, lj = with_logabsdet_jacobian(trf, view(y, r))
        _set_lastindex!(x, idx, xr)
        logjac += lj
    end
    return x, logjac
end
function (t::ProductVecInvTransform{<:AbstractArray})(y::AbstractVector{T}) where {T}
    x = Array{T}(undef, t.base_size..., size(t.transforms)...)
    idxs = _cartesian_indices(t.transforms)
    for (idx, trf, r) in zip(idxs, t.transforms, t.ranges)
        xr = trf(view(y, r))
        _set_lastindex!(x, idx, xr)
    end
    return x
end

@generated function with_logabsdet_jacobian(
    t::ProductVecInvTransform{<:NamedTuple{names}}, y::AbstractVector{T}
) where {names,T}
    expr = Expr(:block)
    push!(expr.args, :(x = (;)))
    push!(expr.args, :(logjac = _fzero($T)))
    for nm in names
        push!(
            expr.args,
            :((xr, lj) = with_logabsdet_jacobian(t.transforms.$nm, view(y, t.ranges.$nm))),
        )
        push!(expr.args, :(x = (x..., $nm=xr)))
        push!(expr.args, :(logjac += lj))
    end
    push!(expr.args, :(return x, logjac))
    return expr
end
@generated function (t::ProductVecInvTransform{<:NamedTuple{names}})(
    y::AbstractVector{T}
) where {names,T}
    expr = Expr(:tuple)
    for nm in names
        push!(expr.args, :($nm = t.transforms.$nm(view(y, t.ranges.$nm))))
    end
    return expr
end

@generated function _make_transform(
    dists::NamedTuple{names}, indiv_transform_fn, length_fn, struct_type
) where {names}
    exprs = []
    trfms = Expr(:tuple)
    for nm in names
        push!(trfms.args, :($nm = indiv_transform_fn(dists.$nm)))
    end
    push!(exprs, :(trfms = $trfms))
    push!(exprs, :(ranges = ()))
    push!(exprs, :(offset = 1))
    for nm in names
        push!(exprs, :(this_length = length_fn(dists.$nm)))
        push!(exprs, :(ranges = (ranges..., $nm=offset:(offset + this_length - 1))))
        push!(exprs, :(offset += this_length))
    end
    push!(exprs, :(return struct_type(trfms, ranges, size(dists[1]))))
    return Expr(:block, exprs...)
end

@generated function _make_transform(
    dists::NTuple{NDists,D.Distribution}, indiv_transform_fn, length_fn, struct_type
) where {NDists}
    exprs = []
    trfms = Expr(:tuple)
    for i in 1:NDists
        push!(trfms.args, :(indiv_transform_fn(dists[$i])))
    end
    push!(exprs, :(trfms = $trfms))
    push!(exprs, :(ranges = ()))
    push!(exprs, :(offset = 1))
    for i in 1:NDists
        push!(exprs, :(this_length = length_fn(dists[$i])))
        push!(exprs, :(ranges = (ranges..., offset:(offset + this_length - 1))))
        push!(exprs, :(offset += this_length))
    end
    push!(exprs, :(return struct_type(trfms, ranges, size(dists[1]))))
    return Expr(:block, exprs...)
end

function _make_transform(
    dists::AbstractArray{<:D.Distribution}, indiv_transform_fn, length_fn, struct_type
)
    trfms = map(indiv_transform_fn, dists)
    ranges = Array{UnitRange{Int}}(undef, size(dists)...)
    offset = 1
    for (i, dist) in enumerate(dists)
        this_length = length_fn(dist)
        ranges[i] = offset:(offset + this_length - 1)
        offset += this_length
    end
    return struct_type(trfms, ranges, size(dists[1]))
end

for (product_type, dist_field) in (
    (D.ProductNamedTupleDistribution, :dists),
    (D.ProductDistribution, :dists),
    # Annoyingly, vectors of univariate distributions become D.Product rather than
    # D.ProductDistribution (which handles all other tuple/arrays).
    (D.Product, :v),
)
    @eval begin
        function from_vec(d::$product_type)
            return _make_transform(
                d.$dist_field, from_vec, vec_length, ProductVecInvTransform
            )
        end
        function from_linked_vec(d::$product_type)
            return _make_transform(
                d.$dist_field, from_linked_vec, linked_vec_length, ProductVecInvTransform
            )
        end
        function to_vec(d::$product_type)
            return _make_transform(d.$dist_field, to_vec, vec_length, ProductVecTransform)
        end
        function to_linked_vec(d::$product_type)
            return _make_transform(
                d.$dist_field, to_linked_vec, linked_vec_length, ProductVecTransform
            )
        end

        vec_length(d::$product_type) = sum(vec_length, d.$dist_field)
        linked_vec_length(d::$product_type) = sum(linked_vec_length, d.$dist_field)
    end
end

"""
Add extra indices to the end of an optic.

For example, if `i` is `(1, 2)`, this does

```julia
x    ->  x[1, 2]
x.a  ->  x.a[1, 2]
x[3] ->  x[3, 1, 2]
```
"""
append_index(::Nothing, i) = nothing
append_index(::AbstractPPL.Iden, i) = @opticof(_[i])
function append_index(p::AbstractPPL.Property{sym}, i) where {sym}
    return AbstractPPL.Property{sym}(append_index(p.child, i))
end
function append_index(p::AbstractPPL.Index, i)
    return if p.child isa AbstractPPL.Iden
        AbstractPPL.Index((p.ix..., i), p.kw, p.child)
    else
        AbstractPPL.Index(p.ix, p.kw, append_index(p.child, i))
    end
end

# Add an extra symbol to the front of an optic.
prepend_symbol(s::Symbol, optic::AbstractPPL.AbstractOptic) = AbstractPPL.Property{s}(optic)
prepend_symbol(::Symbol, nothing) = nothing

for f in (:optic_vec, :linked_optic_vec)
    for (product_type, dist_field) in ((D.Product, :v), (D.ProductDistribution, :dists))
        @eval begin
            function $f(d::$product_type)
                optics = Union{}[]
                idxs = _cartesian_indices(d.$dist_field)
                for (idx, dist) in zip(idxs, d.$dist_field)
                    this_dist_optics = $f(dist)
                    new_optics = map(optic -> append_index(optic, idx), this_dist_optics)
                    optics = vcat(optics, new_optics)
                end
                return optics
            end
        end
    end

    @eval begin
        function $f(d::D.ProductNamedTupleDistribution)
            optics = Union{}[]
            for (nm, dist) in pairs(d.dists)
                this_dist_optics = $f(dist)
                new_optics = map(optic -> prepend_symbol(nm, optic), this_dist_optics)
                optics = vcat(optics, new_optics)
            end
            return optics
        end
    end
end
