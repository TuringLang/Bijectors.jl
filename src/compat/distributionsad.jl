using .DistributionsAD: TuringDirichlet, TuringWishart, TuringInverseWishart,
                        FillVectorOfUnivariate, FillMatrixOfUnivariate,
                        MatrixOfUnivariate, FillVectorOfMultivariate, VectorOfMultivariate,
                        TuringScalMvNormal, TuringDiagMvNormal, TuringDenseMvNormal
using Distributions: AbstractMvLogNormal

# Bijectors

bijector(::TuringDirichlet) = SimplexBijector()
bijector(::TuringWishart) = PDBijector()
bijector(::TuringInverseWishart) = PDBijector()
bijector(::TuringScalMvNormal) = Identity{1}()
bijector(::TuringDiagMvNormal) = Identity{1}()
bijector(::TuringDenseMvNormal) = Identity{1}()

bijector(d::FillVectorOfUnivariate{Continuous}) = up1(bijector(d.v.value))
bijector(d::FillMatrixOfUnivariate{Continuous}) = up1(up1(bijector(d.dists.value)))
bijector(d::MatrixOfUnivariate{Discrete}) = Identity{2}()
bijector(d::MatrixOfUnivariate{Continuous}) = TruncatedBijector{2}(_minmax(d.dists)...)
bijector(d::VectorOfMultivariate{Discrete}) = Identity{2}()
for T in (:VectorOfMultivariate, :FillVectorOfMultivariate)
    @eval begin
        bijector(d::$T{Continuous, <:MvNormal}) = Identity{2}()
        bijector(d::$T{Continuous, <:TuringScalMvNormal}) = Identity{2}()
        bijector(d::$T{Continuous, <:TuringDiagMvNormal}) = Identity{2}()
        bijector(d::$T{Continuous, <:TuringDenseMvNormal}) = Identity{2}()
        bijector(d::$T{Continuous, <:MvNormalCanon}) = Identity{2}()
        bijector(d::$T{Continuous, <:AbstractMvLogNormal}) = Log{2}()
        bijector(d::$T{Continuous, <:SimplexDistribution}) = SimplexBijector{2}()
        bijector(d::$T{Continuous, <:TuringDirichlet}) = SimplexBijector{2}()
    end
end
bijector(d::FillVectorOfMultivariate{Continuous}) = up1(bijector(d.dists.value))

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
