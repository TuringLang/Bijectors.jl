var documenterSearchIndex = {"docs":
[{"location":"transforms/#Usage","page":"Transforms","title":"Usage","text":"","category":"section"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"A very simple example of a \"bijector\"/diffeomorphism, i.e. a differentiable transformation with a differentiable inverse, is the exp function:","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"The inverse of exp is log.\nThe derivative of exp at an input x is simply exp(x), hence logabsdetjac is simply x.","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"using Bijectors\ntransform(exp, 1.0)\nlogabsdetjac(exp, 1.0)\nwith_logabsdet_jacobian(exp, 1.0)","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Some transformations are well-defined for different types of inputs, e.g. exp can also act elementwise on an N-dimensional Array{<:Real,N}. To specify that a transformation should act elementwise, we use the elementwise method:","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"x = ones(2, 2)\ntransform(elementwise(exp), x)\nlogabsdetjac(elementwise(exp), x)\nwith_logabsdet_jacobian(elementwise(exp), x)","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"These methods also work nicely for compositions of transformations:","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"transform(elementwise(log ∘ exp), x)","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Unlike exp, some transformations have parameters affecting the resulting transformation they represent, e.g. Logit has two parameters a and b representing the lower- and upper-bound, respectively, of its domain:","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"using Bijectors: Logit\n\nf = Logit(0.0, 1.0)\nf(rand()) # takes us from `(0, 1)` to `(-∞, ∞)`","category":"page"},{"location":"transforms/#User-facing-methods","page":"Transforms","title":"User-facing methods","text":"","category":"section"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Without mutation:","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"transform\nlogabsdetjac","category":"page"},{"location":"transforms/#Bijectors.transform","page":"Transforms","title":"Bijectors.transform","text":"transform(b, x)\n\nTransform x using b, treating x as a single input.\n\n\n\n\n\n","category":"function"},{"location":"transforms/#Bijectors.logabsdetjac","page":"Transforms","title":"Bijectors.logabsdetjac","text":"logabsdetjac(b, x)\n\nReturn log(abs(det(J(b, x)))), where J(b, x) is the jacobian of b at x.\n\n\n\n\n\n","category":"function"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"with_logabsdet_jacobian","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"With mutation:","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"transform!\nlogabsdetjac!\nwith_logabsdet_jacobian!","category":"page"},{"location":"transforms/#Bijectors.transform!","page":"Transforms","title":"Bijectors.transform!","text":"transform!(b, x[, y])\n\nTransform x using b, storing the result in y.\n\nIf y is not provided, x is used as the output.\n\n\n\n\n\n","category":"function"},{"location":"transforms/#Bijectors.logabsdetjac!","page":"Transforms","title":"Bijectors.logabsdetjac!","text":"logabsdetjac!(b, x[, logjac])\n\nCompute log(abs(det(J(b, x)))) and store the result in logjac, where J(b, x) is the jacobian of b at x.\n\n\n\n\n\n","category":"function"},{"location":"transforms/#Bijectors.with_logabsdet_jacobian!","page":"Transforms","title":"Bijectors.with_logabsdet_jacobian!","text":"with_logabsdet_jacobian!(b, x[, y, logjac])\n\nCompute transform(b, x) and logabsdetjac(b, x), storing the result in y and logjac, respetively.\n\nIf y is not provided, then x will be used in its place.\n\nDefaults to calling with_logabsdet_jacobian(b, x) and updating y and logjac with the result.\n\n\n\n\n\n","category":"function"},{"location":"transforms/#Implementing-a-transformation","page":"Transforms","title":"Implementing a transformation","text":"","category":"section"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Any callable can be made into a bijector by providing an implementation of ChangeOfVariables.with_logabsdet_jacobian(b, x).","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"You can also optionally implement transform and logabsdetjac to avoid redundant computations. This is usually only worth it if you expect transform or logabsdetjac to be used heavily without the other.","category":"page"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Similarly with the mutable versions with_logabsdet_jacobian!, transform!, and logabsdetjac!.","category":"page"},{"location":"transforms/#Working-with-Distributions.jl","page":"Transforms","title":"Working with Distributions.jl","text":"","category":"section"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Bijectors.bijector\nBijectors.transformed(d::Distribution, b::Bijector)","category":"page"},{"location":"transforms/#Bijectors.bijector","page":"Transforms","title":"Bijectors.bijector","text":"bijector(d::Distribution)\n\nReturns the constrained-to-unconstrained bijector for distribution d.\n\n\n\n\n\n","category":"function"},{"location":"transforms/#Bijectors.transformed-Tuple{Distribution, Bijector}","page":"Transforms","title":"Bijectors.transformed","text":"transformed(d::Distribution)\ntransformed(d::Distribution, b::Bijector)\n\nCouples distribution d with the bijector b by returning a TransformedDistribution.\n\nIf no bijector is provided, i.e. transformed(d) is called, then  transformed(d, bijector(d)) is returned.\n\n\n\n\n\n","category":"method"},{"location":"transforms/#Utilities","page":"Transforms","title":"Utilities","text":"","category":"section"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Bijectors.elementwise\nBijectors.isinvertible\nBijectors.isclosedform(t::Bijectors.Transform)","category":"page"},{"location":"transforms/#Bijectors.elementwise","page":"Transforms","title":"Bijectors.elementwise","text":"elementwise(f)\n\nAlias for Base.Fix1(broadcast, f).\n\nIn the case where f::ComposedFunction, the result is Base.Fix1(broadcast, f.outer) ∘ Base.Fix1(broadcast, f.inner) rather than Base.Fix1(broadcast, f).\n\n\n\n\n\n","category":"function"},{"location":"transforms/#Bijectors.isinvertible","page":"Transforms","title":"Bijectors.isinvertible","text":"isinvertible(t)\n\nReturn true if t is invertible, and false otherwise.\n\n\n\n\n\n","category":"function"},{"location":"transforms/#Bijectors.isclosedform-Tuple{Bijectors.Transform}","page":"Transforms","title":"Bijectors.isclosedform","text":"isclosedform(b::Transform)::bool\nisclosedform(b⁻¹::Inverse{<:Transform})::bool\n\nReturns true or false depending on whether or not evaluation of b has a closed-form implementation.\n\nMost transformations have closed-form evaluations, but there are cases where this is not the case. For example the inverse evaluation of PlanarLayer requires an iterative procedure to evaluate.\n\n\n\n\n\n","category":"method"},{"location":"transforms/#API","page":"Transforms","title":"API","text":"","category":"section"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Bijectors.Transform\nBijectors.Bijector\nBijectors.Inverse","category":"page"},{"location":"transforms/#Bijectors.Transform","page":"Transforms","title":"Bijectors.Transform","text":"Abstract type for a transformation.\n\nImplementing\n\nA subtype of Transform of should at least implement transform(b, x).\n\nIf the Transform is also invertible:\n\nRequired:\nEither of the following:\ntransform(::Inverse{<:MyTransform}, x): the transform for its inverse.\nInverseFunctions.inverse(b::MyTransform): returns an existing Transform.\nlogabsdetjac: computes the log-abs-det jacobian factor.\nOptional:\nwith_logabsdet_jacobian: transform and logabsdetjac combined. Useful in cases where we can exploit shared computation in the two.\n\nFor the above methods, there are mutating versions which can optionally be implemented:\n\nwith_logabsdet_jacobian!\nlogabsdetjac!\nwith_logabsdet_jacobian!\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.Bijector","page":"Transforms","title":"Bijectors.Bijector","text":"Abstract type of a bijector, i.e. differentiable bijection with differentiable inverse.\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.Inverse","page":"Transforms","title":"Bijectors.Inverse","text":"inverse(b::Transform)\nInverse(b::Transform)\n\nA Transform representing the inverse transform of b.\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors","page":"Transforms","title":"Bijectors","text":"","category":"section"},{"location":"transforms/","page":"Transforms","title":"Transforms","text":"Bijectors.CorrBijector\nBijectors.LeakyReLU\nBijectors.Stacked\nBijectors.RationalQuadraticSpline\nBijectors.Coupling\nBijectors.OrderedBijector\nBijectors.NamedTransform\nBijectors.NamedCoupling","category":"page"},{"location":"transforms/#Bijectors.CorrBijector","page":"Transforms","title":"Bijectors.CorrBijector","text":"CorrBijector <: Bijector\n\nA bijector implementation of Stan's parametrization method for Correlation matrix: https://mc-stan.org/docs/2_23/reference-manual/correlation-matrix-transform-section.html\n\nBasically, a unconstrained strictly upper triangular matrix y is transformed to  a correlation matrix by following readable but not that efficient form:\n\nK = size(y, 1)\nz = tanh.(y)\n\nfor j=1:K, i=1:K\n    if i>j\n        w[i,j] = 0\n    elseif 1==i==j\n        w[i,j] = 1\n    elseif 1<i==j\n        w[i,j] = prod(sqrt(1 .- z[1:i-1, j].^2))\n    elseif 1==i<j\n        w[i,j] = z[i,j]\n    elseif 1<i<j\n        w[i,j] = z[i,j] * prod(sqrt(1 .- z[1:i-1, j].^2))\n    end\nend\n\nIt is easy to see that every column is a unit vector, for example:\n\nw3' w3 ==\nw[1,3]^2 + w[2,3]^2 + w[3,3]^2 ==\nz[1,3]^2 + (z[2,3] * sqrt(1 - z[1,3]^2))^2 + (sqrt(1-z[1,3]^2) * sqrt(1-z[2,3]^2))^2 ==\nz[1,3]^2 + z[2,3]^2 * (1-z[1,3]^2) + (1-z[1,3]^2) * (1-z[2,3]^2) ==\nz[1,3]^2 + z[2,3]^2 - z[2,3]^2 * z[1,3]^2 + 1 -z[1,3]^2 - z[2,3]^2 + z[1,3]^2 * z[2,3]^2 ==\n1\n\nAnd diagonal elements are positive, so w is a cholesky factor for a positive matrix.\n\nx = w' * w\n\nConsider block matrix representation for x\n\nx = [w1'; w2'; ... wn'] * [w1 w2 ... wn] == \n[w1'w1 w1'w2 ... w1'wn;\n w2'w1 w2'w2 ... w2'wn;\n ...\n]\n\nThe diagonal elements are given by wk'wk = 1, thus x is a correlation matrix.\n\nEvery step is invertible, so this is a bijection(bijector).\n\nNote: The implementation doesn't follow their \"manageable expression\" directly, because their equation seems wrong (7/30/2020). Insteadly it follows definition  above the \"manageable expression\" directly, which is also described in above doc.\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.LeakyReLU","page":"Transforms","title":"Bijectors.LeakyReLU","text":"LeakyReLU{T}(α::T) <: Bijector\n\nDefines the invertible mapping\n\nx ↦ x if x ≥ 0 else αx\n\nwhere α > 0.\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.Stacked","page":"Transforms","title":"Bijectors.Stacked","text":"Stacked(bs)\nStacked(bs, ranges)\nstack(bs::Bijector...)\n\nA Bijector which stacks bijectors together which can then be applied to a vector where bs[i]::Bijector is applied to x[ranges[i]]::UnitRange{Int}.\n\nArguments\n\nbs can be either a Tuple or an AbstractArray of 0- and/or 1-dimensional bijectors\nIf bs is a Tuple, implementations are type-stable using generated functions\nIf bs is an AbstractArray, implementations are not type-stable and use iterative methods\nranges needs to be an iterable consisting of UnitRange{Int}\nlength(bs) == length(ranges) needs to be true.\n\nExamples\n\nb1 = Logit(0.0, 1.0)\nb2 = identity\nb = stack(b1, b2)\nb([0.0, 1.0]) == [b1(0.0), 1.0]  # => true\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.RationalQuadraticSpline","page":"Transforms","title":"Bijectors.RationalQuadraticSpline","text":"RationalQuadraticSpline{T} <: Bijector\n\nImplementation of the Rational Quadratic Spline flow [1].\n\nOutside of the interval [minimum(widths), maximum(widths)], this mapping is given  by the identity map. \nInside the interval it's given by a monotonic spline (i.e. monotonic polynomials  connected at intermediate points) with endpoints fixed so as to continuously transform into the identity map.\n\nFor the sake of efficiency, there are separate implementations for 0-dimensional and 1-dimensional inputs.\n\nNotes\n\nThere are two constructors for RationalQuadraticSpline:\n\nRationalQuadraticSpline(widths, heights, derivatives): it is assumed that widths, \n\nheights, and derivatives satisfy the constraints that makes this a valid bijector, i.e.\n\nwidths: monotonically increasing and length(widths) == K,\nheights: monotonically increasing and length(heights) == K,\nderivatives: non-negative and derivatives[1] == derivatives[end] == 1.\nRationalQuadraticSpline(widths, heights, derivatives, B): other than than the lengths,  no assumptions are made on parameters. Therefore we will transform the parameters s.t.:\nwidths_new ∈ [-B, B]ᴷ⁺¹, where K == length(widths),\nheights_new ∈ [-B, B]ᴷ⁺¹, where K == length(heights),\nderivatives_new ∈ (0, ∞)ᴷ⁺¹ with derivatives_new[1] == derivates_new[end] == 1,  where (K - 1) == length(derivatives).\n\nExamples\n\nUnivariate\n\njulia> using StableRNGs: StableRNG; rng = StableRNG(42);  # For reproducibility.\n\njulia> using Bijectors: RationalQuadraticSpline\n\njulia> K = 3; B = 2;\n\njulia> # Monotonic spline on '[-B, B]' with `K` intermediate knots/\"connection points\".\n       b = RationalQuadraticSpline(randn(rng, K), randn(rng, K), randn(rng, K - 1), B);\n\njulia> b(0.5) # inside of `[-B, B]` → transformed\n1.1943325397834206\n\njulia> b(5.) # outside of `[-B, B]` → not transformed\n5.0\n\njulia> b = RationalQuadraticSpline(b.widths, b.heights, b.derivatives);\n\njulia> b(0.5) # inside of `[-B, B]` → transformed\n1.1943325397834206\n\njulia> d = 2; K = 3; B = 2;\n\njulia> b = RationalQuadraticSpline(randn(rng, d, K), randn(rng, d, K), randn(rng, d, K - 1), B);\n\njulia> b([-1., 1.])\n2-element Vector{Float64}:\n -1.5660106244288925\n  0.5384702734738573\n\njulia> b([-5., 5.])\n2-element Vector{Float64}:\n -5.0\n  5.0\n\njulia> b([-1., 5.])\n2-element Vector{Float64}:\n -1.5660106244288925\n  5.0\n\nReferences\n\n[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). \n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.Coupling","page":"Transforms","title":"Bijectors.Coupling","text":"Coupling{F, M}(θ::F, mask::M)\n\nImplements a coupling-layer as defined in [1].\n\nExamples\n\njulia> using Bijectors: Shift, Coupling, PartitionMask, coupling, couple\n\njulia> m = PartitionMask(3, [1], [2]); # <= going to use x[2] to parameterize transform of x[1]\n\njulia> cl = Coupling(Shift, m); # <= will do `y[1:1] = x[1:1] + x[2:2]`;\n\njulia> x = [1., 2., 3.];\n\njulia> cl(x)\n3-element Vector{Float64}:\n 3.0\n 2.0\n 3.0\n\njulia> inverse(cl)(cl(x))\n3-element Vector{Float64}:\n 1.0\n 2.0\n 3.0\n\njulia> coupling(cl) # get the `Bijector` map `θ -> b(⋅, θ)`\nShift\n\njulia> couple(cl, x) # get the `Bijector` resulting from `x`\nShift([2.0])\n\njulia> with_logabsdet_jacobian(cl, x)\n([3.0, 2.0, 3.0], 0.0)\n\nReferences\n\n[1] Kobyzev, I., Prince, S., & Brubaker, M. A., Normalizing flows: introduction and ideas, CoRR, (),  (2019). \n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.OrderedBijector","page":"Transforms","title":"Bijectors.OrderedBijector","text":"OrderedBijector()\n\nA bijector mapping ordered vectors in ℝᵈ to unordered vectors in ℝᵈ.\n\nSee also\n\nStan's documentation\nNote that this transformation and its inverse are the opposite of in this reference.\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.NamedTransform","page":"Transforms","title":"Bijectors.NamedTransform","text":"NamedTransform <: AbstractNamedTransform\n\nWraps a NamedTuple of key -> Bijector pairs, implementing evaluation, inversion, etc.\n\nExamples\n\njulia> using Bijectors: NamedTransform, Scale\n\njulia> b = NamedTransform((a = Scale(2.0), b = exp));\n\njulia> x = (a = 1., b = 0., c = 42.);\n\njulia> b(x)\n(a = 2.0, b = 1.0, c = 42.0)\n\njulia> (a = 2 * x.a, b = exp(x.b), c = x.c)\n(a = 2.0, b = 1.0, c = 42.0)\n\n\n\n\n\n","category":"type"},{"location":"transforms/#Bijectors.NamedCoupling","page":"Transforms","title":"Bijectors.NamedCoupling","text":"NamedCoupling{target, deps, F} <: AbstractNamedTransform\n\nImplements a coupling layer for named bijectors.\n\nSee also: Coupling\n\nExamples\n\njulia> using Bijectors: NamedCoupling, Scale\n\njulia> b = NamedCoupling(:b, (:a, :c), (a, c) -> Scale(a + c));\n\njulia> x = (a = 1., b = 2., c = 3.);\n\njulia> b(x)\n(a = 1.0, b = 8.0, c = 3.0)\n\njulia> (a = x.a, b = (x.a + x.c) * x.b, c = x.c)\n(a = 1.0, b = 8.0, c = 3.0)\n\n\n\n\n\n","category":"type"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Bijectors","category":"page"},{"location":"examples/#Univariate-ADVI-example","page":"Examples","title":"Univariate ADVI example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"But the real utility of TransformedDistribution becomes more apparent when using transformed(dist, b) for any bijector b. To get the transformed distribution corresponding to the Beta(2, 2), we called transformed(dist) before. This is simply an alias for transformed(dist, bijector(dist)). Remember bijector(dist) returns the constrained-to-constrained bijector for that particular Distribution. But we can of course construct a TransformedDistribution using different bijectors with the same dist. This is particularly useful in something called Automatic Differentiation Variational Inference (ADVI).[2] An important part of ADVI is to approximate a constrained distribution, e.g. Beta, as follows:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Sample x from a Normal with parameters μ and σ, i.e. x ~ Normal(μ, σ).\nTransform x to y s.t. y ∈ support(Beta), with the transform being a differentiable bijection with a differentiable inverse (a \"bijector\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"This then defines a probability density with same support as Beta! Of course, it's unlikely that it will be the same density, but it's an approximation. Creating such a distribution becomes trivial with Bijector and TransformedDistribution:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using StableRNGs: StableRNG\nrng = StableRNG(42);\ndist = Beta(2, 2)\nb = bijector(dist)              # (0, 1) → ℝ\nb⁻¹ = inverse(b)                # ℝ → (0, 1)\ntd = transformed(Normal(), b⁻¹) # x ∼ 𝓝(0, 1) then b(x) ∈ (0, 1)\nx = rand(rng, td)                   # ∈ (0, 1)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"It's worth noting that support(Beta) is the closed interval [0, 1], while the constrained-to-unconstrained bijection, Logit in this case, is only well-defined as a map (0, 1) → ℝ for the open interval (0, 1). This is of course not an implementation detail. ℝ is itself open, thus no continuous bijection exists from a closed interval to ℝ. But since the boundaries of a closed interval has what's known as measure zero, this doesn't end up affecting the resulting density with support on the entire real line. In practice, this means that","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"td = transformed(Beta())\ninverse(td.transform)(rand(rng, td))","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"will never result in 0 or 1 though any sample arbitrarily close to either 0 or 1 is possible. Disclaimer: numerical accuracy is limited, so you might still see 0 and 1 if you're lucky.","category":"page"},{"location":"examples/#Multivariate-ADVI-example","page":"Examples","title":"Multivariate ADVI example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"We can also do multivariate ADVI using the Stacked bijector. Stacked gives us a way to combine univariate and/or multivariate bijectors into a singe multivariate bijector. Say you have a vector x of length 2 and you want to transform the first entry using Exp and the second entry using Log. Stacked gives you an easy and efficient way of representing such a bijector.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Bijectors: SimplexBijector\n\n# Original distributions\ndists = (Beta(), InverseGamma(), Dirichlet(2, 3));\n\n# Construct the corresponding ranges\nranges = [];\nidx = 1;\n\nfor i in 1:length(dists)\n    d = dists[i]\n    push!(ranges, idx:(idx + length(d) - 1))\n\n    global idx\n    idx += length(d)\nend;\n\nranges\n\n# Base distribution; mean-field normal\nnum_params = ranges[end][end]\n\nd = MvNormal(zeros(num_params), ones(num_params));\n\n# Construct the transform\nbs = bijector.(dists);     # constrained-to-unconstrained bijectors for dists\nibs = inverse.(bs);            # invert, so we get unconstrained-to-constrained\nsb = Stacked(ibs, ranges) # => Stacked <: Bijector\n\n# Mean-field normal with unconstrained-to-constrained stacked bijector\ntd = transformed(d, sb);\ny = rand(td)\n0.0 ≤ y[1] ≤ 1.0\n0.0 < y[2]\nsum(y[3:4]) ≈ 1.0","category":"page"},{"location":"examples/#Normalizing-flows","page":"Examples","title":"Normalizing flows","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"A very interesting application is that of normalizing flows.[1] Usually this is done by sampling from a multivariate normal distribution, and then transforming this to a target distribution using invertible neural networks. Currently there are two such transforms available in Bijectors.jl: PlanarLayer and RadialLayer. Let's create a flow with a single PlanarLayer:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Bijectors\nusing StableRNGs: StableRNG\nrng = StableRNG(42);","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"d = MvNormal(zeros(2), ones(2));\nb = PlanarLayer(2)\nflow = transformed(d, b)\nflow isa MultivariateDistribution","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"That's it. Now we can sample from it using rand and compute the logpdf, like any other Distribution.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"y = rand(rng, flow)\nlogpdf(flow, y)         # uses inverse of `b`","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Similarily to the multivariate ADVI example, we could use Stacked to get a bounded flow:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"d = MvNormal(zeros(2), ones(2));\nibs = inverse.(bijector.((InverseGamma(2, 3), Beta())));\nsb = stack(ibs...) # == Stacked(ibs) == Stacked(ibs, [i:i for i = 1:length(ibs)]\nb = sb ∘ PlanarLayer(2)\ntd = transformed(d, b);\ny = rand(rng, td)\n0 < y[1]\n0 ≤ y[2] ≤ 1","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Want to fit the flow?","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using Zygote\n\n# Construct the flow.\nb = PlanarLayer(2)\n\n# Convenient for extracting parameters and reconstructing the flow.\nusing Functors\nθs, reconstruct = Functors.functor(b);\n\n# Make the objective a `struct` to avoid capturing global variables.\nstruct NLLObjective{R,D,T}\n    reconstruct::R\n    basedist::D\n    data::T\nend\n\nfunction (obj::NLLObjective)(θs...)\n    transformed_dist = transformed(obj.basedist, obj.reconstruct(θs))\n    return -sum(Base.Fix1(logpdf, transformed_dist), eachcol(obj.data))\nend\n\n# Some random data to estimate the density of.\nxs = randn(2, 1000);\n\n# Construct the objective.\nf = NLLObjective(reconstruct, MvNormal(2, 1), xs);\n\n# Initial loss.\n@info \"Initial loss: $(f(θs...))\"\n\n# Train using gradient descent.\nε = 1e-3;\nfor i in 1:100\n    ∇s = Zygote.gradient(f, θs...)\n    θs = map(θs, ∇s) do θ, ∇\n        θ - ε .* ∇\n    end\nend\n\n# Final loss\n@info \"Finall loss: $(f(θs...))\"\n\n# Very simple check to see if we learned something useful.\nsamples = rand(transformed(f.basedist, f.reconstruct(θs)), 1000);\nmean(eachcol(samples)) # ≈ [0, 0]\ncov(samples; dims=2)   # ≈ I","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"We can easily create more complex flows by simply doing PlanarLayer(10) ∘ PlanarLayer(10) ∘ RadialLayer(10) and so on.","category":"page"},{"location":"distributions/#Basic-usage","page":"Distributions.jl integration","title":"Basic usage","text":"","category":"section"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"Other than the logpdf_with_trans methods, the package also provides a more composable interface through the Bijector types. Consider for example the one from above with Beta(2, 2).","category":"page"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"julia> using Random;\n       Random.seed!(42);\n\njulia> using Bijectors;\n       using Bijectors: Logit;\n\njulia> dist = Beta(2, 2)\nBeta{Float64}(α=2.0, β=2.0)\n\njulia> x = rand(dist)\n0.36888689965963756\n\njulia> b = bijector(dist) # bijection (0, 1) → ℝ\nLogit{Float64}(0.0, 1.0)\n\njulia> y = b(x)\n-0.5369949942509267","category":"page"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"In this case we see that bijector(d::Distribution) returns the corresponding constrained-to-unconstrained bijection for Beta, which indeed is a Logit with a = 0.0 and b = 1.0. The resulting Logit <: Bijector has a method (b::Logit)(x) defined, allowing us to call it just like any other function. Comparing with the above example, b(x) ≈ link(dist, x). Just to convince ourselves:","category":"page"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"julia> b(x) ≈ link(dist, x)\ntrue","category":"page"},{"location":"distributions/#Transforming-distributions","page":"Distributions.jl integration","title":"Transforming distributions","text":"","category":"section"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"using Bijectors","category":"page"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"We can create a transformed Distribution, i.e. a Distribution defined by sampling from a given Distribution and then transforming using a given transformation:","category":"page"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"dist = Beta(2, 2)      # support on (0, 1)\ntdist = transformed(dist) # support on ℝ\n\ntdist isa UnivariateDistribution","category":"page"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"We can the then compute the logpdf for the resulting distribution:","category":"page"},{"location":"distributions/","page":"Distributions.jl integration","title":"Distributions.jl integration","text":"# Some example values\nx = rand(dist)\ny = tdist.transform(x)\n\nlogpdf(tdist, y)","category":"page"},{"location":"#Bijectors.jl","page":"Home","title":"Bijectors.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package implements a set of functions for transforming constrained random variables (e.g. simplexes, intervals) to Euclidean space. The 3 main functions implemented in this package are the link, invlink and logpdf_with_trans for a number of distributions. The distributions supported are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"RealDistribution: Union{Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal, NormalCanon, TDist},\nPositiveDistribution: Union{BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma, InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull},\nUnitDistribution: Union{Beta, KSOneSided, NoncentralBeta},\nSimplexDistribution: Union{Dirichlet},\nPDMatDistribution: Union{InverseWishart, Wishart}, and\nTransformDistribution: Union{T, Truncated{T}} where T<:ContinuousUnivariateDistribution.","category":"page"},{"location":"","page":"Home","title":"Home","text":"All exported names from the Distributions.jl package are reexported from Bijectors.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Bijectors.jl also provides a nice interface for working with these maps: composition, inversion, etc. The following table lists mathematical operations for a bijector and the corresponding code in Bijectors.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Operation Method Automatic\nb ↦ b⁻¹ inverse(b) ✓\n(b₁, b₂) ↦ (b₁ ∘ b₂) b₁ ∘ b₂ ✓\n(b₁, b₂) ↦ [b₁, b₂] stack(b₁, b₂) ✓\nx ↦ b(x) b(x) ×\ny ↦ b⁻¹(y) inverse(b)(y) ×\nx ↦ log｜det J(b, x)｜ logabsdetjac(b, x) AD\nx ↦ b(x), log｜det J(b, x)｜ with_logabsdet_jacobian(b, x) ✓\np ↦ q := b_* p q = transformed(p, b) ✓\ny ∼ q y = rand(q) ✓\np ↦ b such that support(b_* p) = ℝᵈ bijector(p) ✓\n(x ∼ p, b(x), log｜det J(b, x)｜, log q(y)) forward(q) ✓","category":"page"},{"location":"","page":"Home","title":"Home","text":"In this table, b denotes a Bijector, J(b, x) denotes the Jacobian of b evaluated at x, b_* denotes the push-forward of p by b, and x ∼ p denotes x sampled from the distribution with density p.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The \"Automatic\" column in the table refers to whether or not you are required to implement the feature for a custom Bijector. \"AD\" refers to the fact that it can be implemented \"automatically\" using automatic differentiation.","category":"page"}]
}
