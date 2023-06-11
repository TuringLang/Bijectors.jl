module BijectorsDistributionsADExt

if isdefined(Base, :get_extension)
    using Bijectors
    using Bijectors: LinearAlgebra
    using Bijectors.Distributions: AbstractMvLogNormal
    using DistributionsAD:
        TuringDirichlet,
        TuringWishart,
        TuringInverseWishart,
        FillVectorOfUnivariate,
        FillMatrixOfUnivariate,
        MatrixOfUnivariate,
        FillVectorOfMultivariate,
        VectorOfMultivariate,
        TuringScalMvNormal,
        TuringDiagMvNormal,
        TuringDenseMvNormal
else
    using ..Bijectors
    using ..Bijectors: LinearAlgebra
    using ..Bijectors.Distributions: AbstractMvLogNormal
    using ..DistributionsAD:
        TuringDirichlet,
        TuringWishart,
        TuringInverseWishart,
        FillVectorOfUnivariate,
        FillMatrixOfUnivariate,
        MatrixOfUnivariate,
        FillVectorOfMultivariate,
        VectorOfMultivariate,
        TuringScalMvNormal,
        TuringDiagMvNormal,
        TuringDenseMvNormal
end

# Bijectors

Bijectors.bijector(::TuringDirichlet) = Bijectors.SimplexBijector()
Bijectors.bijector(::TuringWishart) = Bijectors.PDBijector()
Bijectors.bijector(::TuringInverseWishart) = Bijectors.PDBijector()
Bijectors.bijector(::TuringScalMvNormal) = identity
Bijectors.bijector(::TuringDiagMvNormal) = identity
Bijectors.bijector(::TuringDenseMvNormal) = identity

Bijectors.bijector(d::FillVectorOfUnivariate{Continuous}) = elementwise(Bijectors.bijector(d.v.value))
function Bijectors.bijector(d::FillMatrixOfUnivariate{Continuous})
    return elementwise(Bijectors.bijector(d.dists.value))
end
Bijectors.bijector(d::MatrixOfUnivariate{Discrete}) = identity
function Bijectors.bijector(d::MatrixOfUnivariate{Continuous})
    return TruncatedBijectors.Bijector(_minmax(d.dists)...)
end
Bijectors.bijector(d::VectorOfMultivariate{Discrete}) = identity
for T in (:VectorOfMultivariate, :FillVectorOfMultivariate)
    @eval begin
        Bijectors.bijector(d::$T{Continuous,<:MvNormal}) = identity
        Bijectors.bijector(d::$T{Continuous,<:TuringScalMvNormal}) = identity
        Bijectors.bijector(d::$T{Continuous,<:TuringDiagMvNormal}) = identity
        Bijectors.bijector(d::$T{Continuous,<:TuringDenseMvNormal}) = identity
        Bijectors.bijector(d::$T{Continuous,<:MvNormalCanon}) = identity
        Bijectors.bijector(d::$T{Continuous,<:AbstractMvLogNormal}) = Log()
        function Bijectors.bijector(d::$T{Continuous,<:SimplexDistribution})
            return Bijectors.SimplexBijector()
        end
        function Bijectors.bijector(d::$T{Continuous,<:TuringDirichlet})
            return Bijectors.SimplexBijector()
        end
    end
end
function Bijectors.bijector(d::FillVectorOfMultivariate{Continuous})
    return columnwise(Bijectors.bijector(d.dists.value))
end

Bijectors.isdirichlet(::VectorOfMultivariate{Continuous,<:Dirichlet}) = true
Bijectors.isdirichlet(::VectorOfMultivariate{Continuous,<:TuringDirichlet}) = true
Bijectors.isdirichlet(::TuringDirichlet) = true

function Bijectors.link(
    d::TuringDirichlet, x::AbstractVecOrMat{<:Real}, ::Val{proj}=Val(true)
) where {proj}
    return Bijectors.SimplexBijector{proj}()(x)
end

function Bijectors.link_jacobian(
    d::TuringDirichlet, x::AbstractVector{<:Real}, ::Val{proj}=Val(true)
) where {proj}
    return jacobian(Bijectors.SimplexBijector{proj}(), x)
end

function Bijectors.invlink(
    d::TuringDirichlet, y::AbstractVecOrMat{<:Real}, ::Val{proj}=Val(true)
) where {proj}
    return inverse(Bijectors.SimplexBijector{proj}())(y)
end
function Bijectors.invlink_jacobian(
    d::TuringDirichlet, y::AbstractVector{<:Real}, ::Val{proj}=Val(true)
) where {proj}
    return jacobian(inverse(Bijectors.SimplexBijector{proj}()), y)
end

Bijectors.ispd(::TuringWishart) = true
Bijectors.ispd(::TuringInverseWishart) = true
function Bijectors.getlogp(d::TuringWishart, Xcf, X)
    return ((d.df - (size(d, 1) + 1)) * LinearAlgebra.logdet(Xcf) - LinearAlgebra.tr(d.chol \ X)) / 2 +
           d.logc0
end
function Bijectors.getlogp(d::TuringInverseWishart, Xcf, X)
    Ψ = d.S
    return -((d.df + size(d, 1) + 1) * LinearAlgebra.logdet(Xcf) + LinearAlgebra.tr(Xcf \ Ψ)) / 2 +
           d.logc0
end

end
