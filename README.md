# Bijectors.jl

[![Build Status](https://travis-ci.org/TuringLang/Bijectors.jl.svg?branch=master)](https://travis-ci.org/TuringLang/Bijectors.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/mvfs8eio2cscwk1m?svg=true)](https://ci.appveyor.com/project/TuringLang/bijectors-jl)


This package implements a set of functions for transforming constrained random variables (e.g. simplexes, intervals) to Euclidean space. The 3 main functions implemented in this package are the `link`, `invlink` and `logpdf_with_trans` for a number of distributions. The distributions supported are:
1. `RealDistribution`: `Union{Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal, NormalCanon, TDist}`,
2. `PositiveDistribution`: `Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull}`,
3. `UnitDistribution`: `Union{Beta, KSOneSided, NoncentralBeta}`,
4. `SimplexDistribution`: `Union{Dirichlet}`,
5. `PDMatDistribution`: `Union{InverseWishart, Wishart}`, and
6. `TransformDistribution`: `Union{T, Truncated{T}} where T<:ContinuousUnivariateDistribution`.

All exported names from the [Distributions.jl](https://github.com/TuringLang/Bijectors.jl) package are reexported from `Bijectors`.

## Functions

1. `link`: maps a sample of a random distribution `dist` from its support to a value in R^n. Example:

```julia
julia> using Bijectors

julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> x = rand(dist)

0.7472542331020509

julia> y = link(dist, x)
1.084021356473311
```

2. `invlink`: the inverse of the `link` function. Example:

```julia
julia> z = invlink(dist, y)
0.6543406780096065

julia> x == z
true
```

3. `logpdf_with_trans`: finds `log` of the (transformed) probability density function of a distribution `dist` at a sample `x`. Example:

```julia
julia> using Bijectors

julia> dist = Dirichlet(2, 3)
Dirichlet{Float64}(alpha=[3.0, 3.0])

julia> x = rand(dist)
2-element Array{Float64,1}:
 0.46094823621110165
 0.5390517637888984

julia> logpdf_with_trans(dist, x, false) # ignoring the transformation
0.6163709733893024

julia> logpdf_with_trans(dist, x, true) # considering the transformation
-0.7760422307471244
```

## `Bijector` interface
### Basic usage
Other than the `logpdf_with_trans` methods, the package also provides a more composable interface through the `Bijector` types. Consider for example the one from above with `Beta(2, 2)`.

```julia
julia> using Bijectors; using Bijectors: Logit

julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> x = rand(dist)
0.7173326646959575

julia> b = bijector(dist) # bijection (0, 1) → ℝ
Logit{Float64}(0.0, 1.0)

julia> y = b(x)
0.9312689879144197
```

In this case we see that `bijector(d::Distribution)` returns the corresponding constrained-to-unconstrained bijection for `Beta`, which indeed is a `Logit` with `a = 0.0` and `b = 1.0`. The resulting `Logit <: Bijector` has a method `(b::Logit)(x)` defined, allowing us to call it just like any other function. Comparing with the above example, `b(x) == link(dist, x)`. Just to convince ourselves:

```julia
julia> b(x) == link(dist, x)
true
```

#### Inversion

What about `invlink`?

```julia
julia> b⁻¹ = inv(b)
Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0))

julia> b⁻¹(y)
0.7173326646959575

julia> b⁻¹(y) == invlink(dist, y)
true
```

Pretty neat, huh? `Inversed{Logit}` is also a `Bijector` where we've defined `(ib::Inversed{<:Logit})(y)` as the inverse transformation of `(b::Logit)(x)`. Note that it's not always the case that `inv(b) isa Inversed`, e.g. the inverse of `Exp` is simply `Log` so `inv(Exp()) isa Log` is true. 

#### Composition
Also, we can _compose_ bijectors:

```julia
julia> id_y = (b ∘ b⁻¹)
Composed{Tuple{Inversed{Logit{Float64}},Logit{Float64}}}((Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0)), Logit{Float64}(0.0, 1.0)))

julia> id_y(y) ≈ y
true
```
    
And since `Composed isa Bijector`:

```julia
julia> id_x = inv(id_y)
Composed{Tuple{Inversed{Bijectors.Logit{Float64}},Bijectors.Logit{Float64}}}((Inversed{Bijectors.Logit{Float64}}(Bijectors.Logit{Float64}(0.0, 1.0)), Bijectors.Logit{Float64}(0.0, 1.0)))

julia> id_x(x) ≈ x
true
```

#### `logpdf` of `TransformedDistribution`
This far we've seen that we can replicate the functionality provided by `link` and `invlink`. To replicate `logpdf_with_trans` we instead provide a `TransformedDistribution <: Distribution` implementing the `Distribution` interface from Distributions.jl:

```julia
julia> using Bijectors: TransformedDistribution

julia> td = transformed(dist)
TransformedDistribution{Beta{Float64},Logit{Float64},Univariate}(
dist: Beta{Float64}(α=2.0, β=2.0)
transform: Logit{Float64}(0.0, 1.0)
)

julia> td isa UnivariateDistribution
true

julia> logpdf(td, y)
-1.0577727579778098

julia> logpdf_with_trans(dist, x, true)
-1.05777275797781
```

When computing `logpdf(td, y)` where `td` is the _transformed_ distribution corresponding to `Beta(2, 2)`, it makes more semantic sense to compute the pdf of the _transformed_ variable `y` rather than using the "un-transformed" variable `x` to do so, as we do in `logpdf_with_trans`. With that being said, we can also do

```julia
julia> logpdf_forward(td, x)
-1.05777275797781
```

#### `logabsdetjac` and `forward`

In the computation of both `logpdf` and `logpdf_forward` we need to compute `log(abs(det(jacobian(inv(b), y))))` and `log(abs(det(jacobian(b, x))))`, respectively. This computation is available using the `logabsdetjac` method

```julia
julia> logabsdetjac(b⁻¹, y)
-1.595700144883034

julia> logabsdetjac(b, x)
1.595700144883034
```

Notice that

```julia
julia> logabsdetjac(b, x) ≈ -logabsdetjac(b⁻¹, y)
true
```

which is always the case for a differentiable bijection with differentiable inverse. Therefore if you want to compute `logabsdetjac(b⁻¹, y)` and we know that `logabsdetjac(b, b⁻¹(y))` is actually more efficient, we'll return `-logabsdetjac(b, b⁻¹(y))` instead. For some bijectors it might be easy to compute, say, the forward pass `b(x)`, but expensive to compute `b⁻¹(y)`. Because of this you might want to avoid doing anything "backwards", i.e. using `b⁻¹`. This is where `forward` comes to good use:

```julia
julia> forward(b, x)
(rv = 0.9312689879144197, logabsdetjac = 1.595700144883034)
```

Similarily

```julia
julia> forward(inv(b), y)
(rv = 0.7173326646959575, logabsdetjac = -1.595700144883034)
```

In fact, the purpose of `forward` is to just _do the right thing_, not necessarily "forward". In this function we'll have access to both the original value `x` and the transformed value `y`, so we can compute `logabsdetjac(b, x)` in either direction. Furthermore, in a lot of cases we can re-use a lot of the computation from `b(x)` in the computation of `logabsdetjac(b, x)`, or vice-versa. `forward(b, x)` will take advantage of such opportunities (if implemented).

#### Sampling from `TransformedDistribution`
At this point we've only shown that we can replicate the existing functionality. But we said `TransformedDistribution isa Distribution`, so we also have `rand`:

```julia
julia> y = rand(td)              # ∈ ℝ
-0.5231573469209508

julia> x = inv(td.transform)(y)  # transform back to interval [0, 1]
0.37211423725902915
```

This can be quite convenient if you have computations assuming input to be on the real line.

#### Univariate ADVI example
But the real utility of `TransformedDistribution` becomes more apparent when using `transformed(dist, b)` for any bijector `b`. To get the transformed distribution corresponding to the `Beta(2, 2)`, we called `transformed(dist)` before. This is simply an alias for `transformed(dist, bijector(dist))`. Remember `bijector(dist)` returns the constrained-to-constrained bijector for that particular `Distribution`. But we can of course construct a `TransformedDistribution` using different bijectors with the same `dist`. This is particularly useful in something called _Automatic Differentiation Variational Inference (ADVI)_.[2] An important part of ADVI is to approximate a constrained distribution, e.g. `Beta`, as follows:
1. Sample `x` from a `Normal` with parameters `μ` and `σ`, i.e. `x ~ Normal(μ, σ)`.
2. Transform `x` to `y` s.t. `y ∈ support(Beta)`, with the transform being a differentiable bijection with a differentiable inverse (a "bijector")

This then defines a probability density with same _support_ as `Beta`! Of course, it's unlikely that it will be the same density, but it's an _approximation_. Creating such a distribution becomes trivial with `Bijector` and `TransformedDistribution`:

```julia
julia> dist = Beta(2, 2)
Beta{Float64}(α=2.0, β=2.0)

julia> b = bijector(dist)              # (0, 1) → ℝ

julia> b⁻¹ = inv(b)                    # ℝ → (0, 1)
Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0))

julia> td = transformed(Normal(), b⁻¹) # x ∼ 𝓝(0, 1) then b(x) ∈ (0, 1)
TransformedDistribution{Normal{Float64},Inversed{Logit{Float64}},Univariate}(
dist: Normal{Float64}(μ=0.0, σ=1.0)
transform: Inversed{Logit{Float64}}(Logit{Float64}(0.0, 1.0))
)

julia> x = rand(td)                    # ∈ (0, 1)
0.37786466412061664
```

It's worth noting that `support(Beta)` is the _closed_ interval `[0, 1]`, while the constrained-to-unconstrained bijection, `Logit` in this case, is only well-defined as a map `(0, 1) → ℝ` for the _open_ interval `(0, 1)`. This is of course not an implementation detail. `ℝ` is itself open, thus no continuous bijection exists from a _closed_ interval to `ℝ`. But since the boundaries of a closed interval has what's known as measure zero, this doesn't end up affecting the resulting density with support on the entire real line. In practice, this means that

```julia
td = transformed(Beta())

inv(td.transform)(rand(td))
```

will never result in `0` or `1` though any sample arbitrarily close to either `0` or `1` is possible. _Disclaimer: numerical accuracy is limited, so you might still see `0` and `1` if you're lucky._

### Normalizing flows
A very interesting application is that of _normalizing flows_.[1] Usually this is done by sampling from a multivariate normal distribution, and then transforming this to a target distribution using invertible neural networks. Currently there are two such transforms available in Bijectors.jl: `PlanarFlow` and `RadialFlow`. Let's create a flow with a single `PlanarLayer`:

```julia
julia> d = MvNormal(zeros(2), ones(2));

julia> b = PlanarLayer(2)
PlanarLayer{Array{Float64,2},Array{Float64,1}}([1.25544; -0.644276], [0.735741; 0.522381], [-1.19838])

julia> flow = transformed(d, b)
TransformedDistribution{MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Array{Float64,1}},PlanarLayer{Array{Float64,2},Array{Float64,1}},Multivariate}(
dist: DiagNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0 0.0; 0.0 1.0]
)

transform: PlanarLayer{Array{Float64,2},Array{Float64,1}}([1.25544; -0.644276], [0.735741; 0.522381], [-1.19838])
)


julia> flow isa MultivariateDistribution
true
```

That's it. Now we can sample from it using `rand` and compute the `logpdf`, like any other `Distribution`.

```julia
julia> y = rand(flow)
2-element Array{Float64,1}:
 0.8356896540230636 
 0.07708282276548209

julia> logpdf(flow, y)         # uses inverse of `b`; not very efficient for `PlanarFlow` and not 100% accurate
-2.151503833297053

julia> x = rand(flow.dist)
2-element Array{Float64,1}:
 0.8186517293759961 
 0.31896083550211446

julia> logpdf_forward(flow, x) # more efficent and accurate
-2.2489445532797867
```

Want to fit the flow?

```julia
julia> using Tracker

julia> b = PlanarLayer(2, param)                  # construct parameters using `param`
PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}}([0.100896; -0.753183] (tracked), [0.320337; 0.674077] (tracked), [-1.02852] (tracked))

julia> flow = transformed(d, b)
TransformedDistribution{MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Array{Float64,1}},PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}},Multivariate}(
dist: DiagNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0 0.0; 0.0 1.0]
)

transform: PlanarLayer{TrackedArray{…,Array{Float64,2}},TrackedArray{…,Array{Float64,1}}}([0.100896; -0.753183] (tracked), [0.320337; 0.674077] (tracked), [-1.02852] (tracked))
)


julia> rand(flow)
Tracked 2-element Array{Float64,1}:
  0.32015420426554175
 -0.9860754227482333 

julia> x = rand(flow.dist)
2-element Array{Float64,1}:
  0.11278529997563423
 -1.6565063910085815 

julia> Tracker.back!(logpdf_forward(flow, x), 1.0) # backprob

julia> Tracker.grad(b.w)
2×1 Array{Float64,2}:
 -0.277554258517636  
  0.24043919425701835
```

We can easily create more complex flows by simply doing `PlanarFlow(10) ∘ PlanarFlow(10) ∘ RadialFlow(10)` and so on.

In those cases, it might be useful to use Flux.jl's `treelike` to extract the parameters:
```julia
julia> using Flux

julia> @Flux.treelike Composed

julia> @Flux.treelike TransformedDistribution

julia> @Flux.treelike PlanarLayer

julia> Flux.params(flow)
Params([[0.100896; -0.753183] (tracked), [0.320337; 0.674077] (tracked), [-1.02852] (tracked)])
```
Though we might just do this for you in the future, so then all you'll have to do is call `Flux.params`.

Another useful function is the `forward(d::Distribution)` method. It is similar to `forward(b::Bijector)` in the sense that it does a forward pass of the entire process "sample then transform" and returns all the most useful quantities in process using the most efficent computation path.

```julia
julia> x, y, logjac, logpdf_y = forward(flow) # sample + transform and returns all the useful quantities in one pass
(x = [-0.387191, 0.761807], y = [-0.677683, 0.0866711] (tracked), logabsdetjac = -0.07475475048737289 (tracked), logpdf = -2.1282560611425447 (tracked))
```

This method is for example useful when computing quantities such as the _expected lower bound (ELBO)_ between this transformed distribution and some other joint density. If no analytical expression is available, we have to approximate the ELBO by a Monte Carlo estimate. But one term in the ELBO is the entropy of the base density, which we _do_ know analytically in this case. Using the analytical expression for the entropy and then using a monte carlo estimate for the rest of the terms in the ELBO gives an estimate with lower variance than if we used the monte carlo estimate for the entire expectation.

### TODO Normalizing flows with constrained supports
Requires PR with `Stacked` merged.

## Implementing your own `Bijector`
There's mainly two ways you can implement your own `Bijector`, and which way you choose mainly depends on the following question: are you bothered enough to manually implement `logabsdetjac`? If the answer is "Yup!", then you subtype from `Bijector`, if "Naaaah" then you subtype `ADBijector`.

### `<:Bijector`
Here's a simple example taken from the source code, the `Identity`:

```julia
struct Identity <: Bijector end
(::Identity)(x) = x                           # transform itself, "forward"
(::Inversed{<: Identity})(y) = y              # inverse tramsform, "backward"

logabsdetjac(::Identity, y) = zero(eltype(y)) # ∂ₓid(x) = ∂ₓ x = 1 → log(abs(1)) = log(1) = 0
```

A slightly more complex example is `Logit`:

```julia
using StatsFuns: logit, logistic

struct Logit{T<:Real} <: Bijector
    a::T
    b::T
end

(b::Logit)(x) = @. logit((x - b.a) / (b.b - b.a))
(ib::Inversed{<:Logit})(y) = @. (ib.orig.b - ib.orig.a) * logistic(y) + ib.orig.a  # `orig` contains the `Bijector` which was inverted

logabsdetjac(b::Logit, x) = @. - log((x - b.a) * (b.b - x) / (b.b - b.a))
```

(Batch computation is not fully supported by all bijectors yet (see issue #35), but is actively worked on. In the particular case of `Logit` there's only one thing that makes sense, which is elementwise application. Therefore we've added `@.` to the implementation above, thus this works for any `AbstractArray{<:Real}`.)

Then

```julia
julia> b = Logit(0.0, 1.0)
Logit{Float64}(0.0, 1.0)

julia> b(0.6)
0.4054651081081642

julia> inv(b)(y)
0.6

julia> logabsdetjac(b, 0.6)
1.4271163556401458

julia> logabsdetjac(inv(b), y) # defaults to `- logabsdetjac(b, inv(b)(x))`
-1.4271163556401458

julia> forward(b, 0.6)         # defaults to `(rv=b(x), logabsdetjac=logabsdetjac(b, x))`
(rv = 0.4054651081081642, logabsdetjac = 1.4271163556401458)
```

For further efficiency, one could manually implement `forward(b::Logit, x)`:

```julia
julia> import Bijectors: forward, Logit

julia> function forward(b::Logit{<:Real}, x)
           totally_worth_saving = @. (x - b.a) / (b.b - b.a)  # spoiler: it's probably not
           y = logit.(totally_worth_saving)
           logjac = @. - log((b.b - x) * totally_worth_saving)
           return (rv=y, logabsdetjac = logjac)
       end
forward (generic function with 16 methods)

julia> forward(b, 0.6)
(rv = 0.4054651081081642, logabsdetjac = 1.4271163556401458)

julia> @which forward(b, 0.6)
forward(b::Logit{#s4} where #s4<:Real, x) in Main at REPL[43]:2
```

As you can see it's a very contrived example, but you get the idea.

### `<:ADBijector`

We could also have implemented `Logit` as an `ADBijector`:

```julia
using StatsFuns: logit, logistic
using Bijectors: ADBackend

struct ADLogit{T, AD} <: ADBijector{AD}
    a::T
    b::T
end

# ADBackend() returns ForwardDiffAD, which means we use ForwardDiff.jl for AD
ADLogit(a::T, b::T) where {T<:Real} = ADLogit{T, ADBackend()}(a, b)

(b::ADLogit)(x) = @. logit((x - b.a) / (b.b - b.a))
(ib::Inversed{<:ADLogit{<:Real}})(y) = @. (ib.orig.b - ib.orig.a) * logistic(y) + ib.orig.a
```

No implementation of `logabsdetjac`, but:

```julia
julia> b_ad = ADLogit(0.0, 1.0)
ADLogit{Float64,Bijectors.ForwardDiffAD}(0.0, 1.0)

julia> logabsdetjac(b_ad, 0.6)
1.4271163556401458

julia> y = b_ad(0.6)
0.4054651081081642

julia> inv(b_ad)(y)
0.6

julia> logabsdetjac(inv(b_ad), y)
-1.4271163556401458
```

Neat! And just to verify that everything works:

```julia
julia> b = Logit(0.0, 1.0)
Logit{Float64}(0.0, 1.0)

julia> logabsdetjac(b, 0.6)
1.4271163556401458

julia> logabsdetjac(b_ad, 0.6) ≈ logabsdetjac(b, 0.6)
true
```

We can also use Tracker.jl for the AD, rather than ForwardDiff.jl:

```julia
julia> Bijectors.setadbackend(:reverse_diff)
:reverse_diff

julia> b_ad = ADLogit(0.0, 1.0)
ADLogit{Float64,Bijectors.TrackerAD}(0.0, 1.0)

julia> logabsdetjac(b_ad, 0.6)
1.4271163556401458
```


### Reference
Most of the methods and types mention below will have docstrings with more elaborate explanation and examples, e.g.
```julia
help?> Bijectors.Composed
  Composed(ts::A)

  ∘(b1::Bijector, b2::Bijector)::Composed{<:Tuple}
  composel(ts::Bijector...)::Composed{<:Tuple}
  composer(ts::Bijector...)::Composed{<:Tuple}

  A Bijector representing composition of bijectors. composel and composer results in a Composed for which application occurs from left-to-right and right-to-left, respectively.

  Note that all the propsed ways of constructing a Composed returns a Tuple of bijectors. This ensures type-stability of implementations of all relating methdos, e.g. inv.

  Examples
  ≡≡≡≡≡≡≡≡≡≡

  It's important to note that ∘ does what is expected mathematically, which means that the bijectors are applied to the input right-to-left, e.g. first applying b2 and then b1:

  (b1 ∘ b2)(x) == b1(b2(x))     # => true

  But in the Composed struct itself, we store the bijectors left-to-right, so that

  cb1 = b1 ∘ b2                  # => Composed.ts == (b2, b1)
  cb2 = composel(b2, b1)         # => Composed.ts == (b2, b1)
  cb1(x) == cb2(x) == b1(b2(x))  # => true
```
If anything is lacking or not clear in docstrings, feel free to open an issue or PR.

#### Types
The following are the bijectors available:
- Abstract:
  - `Bijector`: super-type of all bijectors. 
  - `ADBijector{AD} <: Bijector`: subtypes of this only require the user to implement `(b::UserBijector)(x)` and `(ib::Inversed{<:UserBijector})(y)`. Automatic differentation will be used to compute the `jacobian(b, x)` and thus `logabsdetjac(b, x).
- Concrete:
  - `Composed`: represents a composition of bijectors.
  - `Identity`: does what it says, i.e. nothing.
  - `Logit`
  - `Exp`
  - `Log`
  - `Scale`: scaling by scalar value, though at the moment only well-defined `logabsdetjac` for univariate. 
  - `Shift`: shifts by a scalar value.
  - `SimplexBijector`: mostly used as the constrained-to-unconstrained bijector for `SimplexDistribution`, e.g. `Dirichlet`.
  - `PlanarLayer`: §4.1 Eq. (10) in [1]
  - `RadialLayer`: §4.1 Eq. (14) in [1]

The distribution interface consists of:
- `TransformedDistribution <: Distribution`: implements the `Distribution` interface from Distributions.jl. This means `rand` and `logpdf` are provided at the moment.

#### Methods
The following methods are implemented by all subtypes of `Bijector`, this also includes bijectors such as `Composed`.
- `(b::Bijector)(x)`: implements the transform of the `Bijector`
- `inv(b::Bijector)`: returns the inverse of `b`, i.e. `ib::Bijector` s.t. `(ib ∘ b)(x) ≈ x`. In most cases this is `Inversed{<:Bijector}`.
- `logabsdetjac(b::Bijector, x)`: computes log(abs(det(jacobian(b, x)))).
- `forward(b::Bijector, x)`: returns named tuple `(rv=b(x), logabsdetjac=logabsdetjac(b, x))` in the most efficient manner.
- `∘`, `composel`, `composer`: convenient and type-safe constructors for `Composed`. `composel(bs...)` composes s.t. the resulting composition is evaluated left-to-right, while `composer(bs...)` is evaluated right-to-left. `∘` is right-to-left, as excepted from standard mathematical notation.
- `jacobian(b::Bijector, x)` [OPTIONAL]: returns the jacobian of the transformation. In some cases the analytical jacobian has been implemented for efficiency.

For `TransformedDistribution`, together with default implementations for `Distribution`, we have the following methods:
- `bijector(d::Distribution)`: returns the default constrained-to-unconstrained bijector for `d`
- `transformed(d::Distribution)`, `transformed(d::Distribution, b::Bijector)`: constructs a `TransformedDistribution` from `d` and `b`.
- `logpdf_forward(d::Distribution, x)`, `logpdf_forward(d::Distribution, x, logjac)`: computes the `logpdf(td, td.transform(x))` using the forward pass, which is potentially faster depending on the transform at hand.
- `forward(d::Distribution)`: returns `(x = rand(dist), y = b(x), logabsdetjac = logabsdetjac(b, x), logpdf = logpdf_forward(td, x))` where `b = td.transform`. This combines sampling from base distribution and transforming into one function. The intention is that this entire process should be performed in the most efficient manner, e.g. the `logabsdetjac(b, x)` call might instead be implemented as `- logabsdetjac(inv(b), b(x))` depending on which is most efficient.

# Bibliography
1. Rezende, D. J., & Mohamed, S., Variational Inference With Normalizing Flows, CoRR, (),  (2015). 
2. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M., Automatic Differentiation Variational Inference, CoRR, (),  (2016).
