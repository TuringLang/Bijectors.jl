using .DistributionsAD: TuringDirichlet, TuringWishart, TuringInverseWishart,
                        FillVectorOfUnivariate, FillMatrixOfUnivariate,
                        MatrixOfUnivariate, FillVectorOfMultivariate, VectorOfMultivariate,
                        TuringScalMvNormal, TuringDiagMvNormal, TuringDenseMvNormal
using Distributions: AbstractMvLogNormal

# Bijectors

bijector(::TuringDirichlet) = SimplexBijector()
bijector(::TuringWishart) = PDBijector()
bijector(::TuringInverseWishart) = PDBijector()
bijector(::TuringScalMvNormal) = Identity()
bijector(::TuringDiagMvNormal) = Identity()
bijector(::TuringDenseMvNormal) = Identity()

bijector(d::FillVectorOfUnivariate{Continuous}) = bijector(d.v.value)
bijector(d::FillMatrixOfUnivariate{Continuous}) = up1(bijector(d.dists.value))
bijector(d::MatrixOfUnivariate{Discrete}) = Identity()
bijector(d::MatrixOfUnivariate{Continuous}) = TruncatedBijector(_minmax(d.dists)...)
bijector(d::VectorOfMultivariate{Discrete}) = Identity()
for T in (:VectorOfMultivariate, :FillVectorOfMultivariate)
    @eval begin
        bijector(d::$T{Continuous, <:MvNormal}) = Identity()
        bijector(d::$T{Continuous, <:TuringScalMvNormal}) = Identity()
        bijector(d::$T{Continuous, <:TuringDiagMvNormal}) = Identity()
        bijector(d::$T{Continuous, <:TuringDenseMvNormal}) = Identity()
        bijector(d::$T{Continuous, <:MvNormalCanon}) = Identity()
        bijector(d::$T{Continuous, <:AbstractMvLogNormal}) = Log()
        bijector(d::$T{Continuous, <:SimplexDistribution}) = SimplexBijector()
        bijector(d::$T{Continuous, <:TuringDirichlet}) = SimplexBijector()
    end
end
bijector(d::FillVectorOfMultivariate{Continuous}) = bijector(d.dists.value)

isdirichlet(::VectorOfMultivariate{Continuous, <:Dirichlet}) = true
isdirichlet(::VectorOfMultivariate{Continuous, <:TuringDirichlet}) = true
isdirichlet(::TuringDirichlet) = true

function link(
    d::TuringDirichlet,
    x::AbstractVecOrMat{<:Real},
    ::Val{proj}=Val(true),
) where {proj}
    return SimplexBijector{proj}()(x)
end

function link_jacobian(
    d::TuringDirichlet,
    x::AbstractVector{<:Real},
    ::Val{proj}=Val(true),
) where {proj}
    return jacobian(SimplexBijector{proj}(), x)
end

function invlink(
    d::TuringDirichlet,
    y::AbstractVecOrMat{<:Real},
    ::Val{proj}=Val(true),
) where {proj}
    return inverse(SimplexBijector{proj}())(y)
end
function invlink_jacobian(
    d::TuringDirichlet,
    y::AbstractVector{<:Real},
    ::Val{proj}=Val(true),
) where {proj}
    return jacobian(inverse(SimplexBijector{proj}()), y)
end

ispd(::TuringWishart) = true
ispd(::TuringInverseWishart) = true
function getlogp(d::TuringWishart, Xcf, X)
    return ((d.df - (size(d, 1) + 1)) * logdet(Xcf) - tr(d.chol \ X)) / 2 + d.logc0
end
function getlogp(d::TuringInverseWishart, Xcf, X)
    Ψ = d.S
    return -((d.df + size(d, 1) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) / 2 + d.logc0
end
