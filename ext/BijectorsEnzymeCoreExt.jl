module BijectorsEnzymeCoreExt

if isdefined(Base, :get_extension)
    using EnzymeCore:
        Active,
        Const,
        Duplicated,
        DuplicatedNoNeed,
        BatchDuplicated,
        BatchDuplicatedNoNeed,
        EnzymeRules
    using Bijectors: find_alpha
else
    using ..EnzymeCore:
        Active,
        Const,
        Duplicated,
        DuplicatedNoNeed,
        BatchDuplicated,
        BatchDuplicatedNoNeed,
        EnzymeRules
    using ..Bijectors: find_alpha
end

# Compute a tuple of partial derivatives wrt non-`Const` arguments
# and `nothing`s for `Const` arguments
function ∂find_alpha(
    Ω::Real,
    wt_y::Union{Const,Active,Duplicated,BatchDuplicated},
    wt_u_hat::Union{Const,Active,Duplicated,BatchDuplicated},
    b::Union{Const,Active,Duplicated,BatchDuplicated},
)
    # We reuse the following term in the computation of the derivatives
    Ωpb = Ω + b.val
    c = wt_u_hat.val * sech(Ωpb)^2
    cp1 = c + 1

    ∂Ω_∂wt_y = wt_y isa Const ? nothing : oneunit(wt_y.val) / cp1
    ∂Ω_∂wt_u_hat = wt_u_hat isa Const ? nothing : -tanh(Ωpb) / cp1
    ∂Ω_∂b = b isa Const ? nothing : -c / cp1

    return (∂Ω_∂wt_y, ∂Ω_∂wt_u_hat, ∂Ω_∂b)
end

# `muladd` for partial derivatives that can deal with `nothing` derivatives
_muladd_partial(::Nothing, ::Const, x::Union{Real,Tuple{Vararg{Real}},Nothing}) = x
_muladd_partial(x::Real, y::Duplicated, z::Real) = muladd(x, y.dval, z)
_muladd_partial(x::Real, y::Duplicated, ::Nothing) = x * y.dval
function _muladd_partial(x::Real, y::BatchDuplicated{<:Real,N}, z::NTuple{N,Real}) where {N}
    let x = x
        map((a, b) -> muladd(x, a, b), y.dval, z)
    end
end
_muladd_partial(x::Real, y::BatchDuplicated, ::Nothing) = map(Base.Fix1(*, x), y.dval)

function EnzymeRules.forward(
    ::Const{typeof(find_alpha)},
    ::Type{RT},
    wt_y::Union{Const,Duplicated,BatchDuplicated},
    wt_u_hat::Union{Const,Duplicated,BatchDuplicated},
    b::Union{Const,Duplicated,BatchDuplicated},
) where {RT<:Union{Const,Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed}}
    # Check that the types of the activities are consistent
    if !(
        RT <: Union{Const,Duplicated,DuplicatedNoNeed} &&
        wt_y isa Union{Const,Duplicated} &&
        wt_u_hat isa Union{Const,Duplicated} &&
        b isa Union{Const,Duplicated}
    ) && !(
        RT <: Union{Const,BatchDuplicated,BatchDuplicatedNoNeed} &&
        wt_y isa Union{Const,BatchDuplicated} &&
        wt_u_hat isa Union{Const,BatchDuplicated} &&
        b isa Union{Const,BatchDuplicated}
    )
        throw(ArgumentError("inconsistent activities"))
    end

    # Compute primal value
    Ω = find_alpha(wt_y.val, wt_u_hat.val, b.val)

    # Early exit if no derivatives are requested
    if RT <: Const
        return Ω
    end

    Ω̇ = if wt_y isa Const && wt_u_hat isa Const && b isa Const
        # Trivial case: All partial derivatives are 0
        zero(Ω)
    else
        # In all other cases we have to compute the partial derivatives
        ∂Ω_∂wt_y, ∂Ω_∂wt_u_hat, ∂Ω_∂b = ∂find_alpha(Ω, wt_y, wt_u_hat, b)
        _muladd_partial(
            ∂Ω_∂wt_y,
            wt_y,
            _muladd_partial(∂Ω_∂wt_u_hat, wt_u_hat, _muladd_partial(∂Ω_∂b, b, nothing)),
        )
    end

    if RT <: Duplicated
        @assert Ω̇ isa Real
        return Duplicated(Ω, Ω̇)
    elseif RT <: DuplicatedNoNeed
        @assert Ω̇ isa Real
        return Ω̇
    elseif RT <: BatchDuplicated
        @assert Ω̇ isa Tuple{Vararg{Real}}
        return BatchDuplicated(Ω, Ω̇)
    else
        @assert RT <: BatchDuplicatedNoNeed
        @assert Ω̇ isa Tuple{Vararg{Real}}
        return Ω̇
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.Config,
    ::Const{typeof(find_alpha)},
    ::Type{RT},
    wt_y::Union{Const,Active},
    wt_u_hat::Union{Const,Active},
    b::Union{Const,Active},
) where {RT<:Union{Const,Active}}
    # Only compute the the original return value if it is actually needed
    Ω =
        if EnzymeRules.needs_primal(config) ||
            EnzymeRules.needs_shadow(config) ||
            !(RT <: Const || (wt_y isa Const && wt_u_hat isa Const && b isa Const))
            find_alpha(wt_y.val, wt_u_hat.val, b.val)
        else
            nothing
        end

    tape = if RT <: Const || (wt_y isa Const && wt_u_hat isa Const && b isa Const)
        # Trivial case: No differentiation or all derivatives are 0
        # Thus no tape is needed
        nothing
    else
        # Derivatives with respect to at least one argument needed
        # They are computed in the reverse pass, and therefore the original return is cached
        # In principle, the partial derivatives could be computed here and be cached
        # But Enzyme only executes the reverse pass once,
        # thus this would not increase efficiency but instead more values would have to be cached
        Ω
    end

    # Ensure that we follow the interface requirements of `augmented_primal`
    primal = EnzymeRules.needs_primal(config) ? Ω : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(Ω) : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    config::EnzymeRules.ConfigWidth{1},
    ::Const{typeof(find_alpha)},
    ::Type{<:Const},
    ::Nothing,
    wt_y::Union{Const,Active},
    wt_u_hat::Union{Const,Active},
    b::Union{Const,Active},
)
    # Trivial case: Nothing to be differentiated (return activity is `Const`)
    return map(x -> x isa Active ? zero(x.val) : nothing, (wt_y, wt_u_hat, b))
end
function EnzymeRules.reverse(
    ::EnzymeRules.ConfigWidth{1},
    ::Const{typeof(find_alpha)},
    ::Active,
    ::Nothing,
    ::Const,
    ::Const,
    ::Const,
)
    # Trivial case: Tape does not exist sice all partial derivatives are 0
    return (nothing, nothing, nothing)
end
function EnzymeRules.reverse(
    ::EnzymeRules.ConfigWidth{1},
    ::Const{typeof(find_alpha)},
    ΔΩ::Active,
    Ω::Real,
    wt_y::Union{Const,Active},
    wt_u_hat::Union{Const,Active},
    b::Union{Const,Active},
)
    # Tape must be `nothing` if all arguments are `Const`
    @assert !(wt_y isa Const && wt_u_hat isa Const && b isa Const)

    # Compute partial derivatives
    ∂Ω_∂xs = ∂find_alpha(Ω, wt_y, wt_u_hat, b)
    let ΔΩ_val = ΔΩ.val
        return map(∂Ω_∂x -> ∂Ω_∂x === nothing ? nothing : ∂Ω_∂x * ΔΩ_val, ∂Ω_∂xs)
    end
end

end  # module
