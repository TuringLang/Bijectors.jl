# Example: Normalizing flows

A very interesting application of bijectors is in _normalizing flows_.
Usually this is done by sampling from a multivariate normal distribution, and then transforming this to a target distribution using invertible neural networks.
Currently there are two such transforms available in Bijectors.jl: `PlanarLayer` and `RadialLayer`.
Let's create a flow with a single `PlanarLayer`:

```@example normalizing-flows
using Bijectors
using StableRNGs: StableRNG
rng = StableRNG(42)

d = MvNormal(zeros(2), ones(2))
b = PlanarLayer(2)
flow = transformed(d, b)
```

`flow` is itself a multivariate distribution, so we can sample from it using `rand` and compute the `logpdf`, like any other `Distribution`.

```@example normalizing-flows
y = rand(rng, flow)
logpdf(flow, y)         # uses inverse of `b`
```

Similarily to the multivariate ADVI example, we could use `Stacked` to get a _bounded_ flow:

```@example normalizing-flows
d = MvNormal(zeros(2), ones(2));
ibs = inverse.(bijector.((InverseGamma(2, 3), Beta())));
sb = Stacked(ibs) # == Stacked(ibs, [i:i for i = 1:length(ibs)]
b = sb ∘ PlanarLayer(2)
td = transformed(d, b);
y = rand(rng, td)
```

(As required, we have that `0 < y[1]` and `0 ≤ y[2] ≤ 1`.)

To fit the flow, we can define an objective function that computes the negative log-likelihood of some data.
We will need to use automatic differentiation to compute gradients of the objective with respect to the parameters.
Since most AD packages require vectorised inputs, this means we also need a way to convert between the vectorised parameters and the `PlanarLayer` struct.

```@example normalizing-flows
using ForwardDiff

# Construct the flow.
b = PlanarLayer(2)

# Obtain a vectorised version of the parameters.
xs_init = vcat(b.w, b.u, b.b)

# Function to reconstruct the `PlanarLayer` from vectorised parameters.
function reconstruct_planarlayer(xs::AbstractVector)
    dim = 2
    w = xs[1:dim]
    u = xs[(dim + 1):(2 * dim)]
    b = xs[end:end]
    return PlanarLayer(w, u, b)
end

# Check that the reconstruction does work...
reconstruct_planarlayer(xs_init) == b
```

Here is the objective function:

```@example normalizing-flows
# Make the objective a `struct` to avoid capturing global variables.
struct NLLObjective{R,D,T}
    reconstruct::R
    basedist::D
    data::T
end

function (obj::NLLObjective)(xs::AbstractVector)
    transformed_dist = transformed(obj.basedist, obj.reconstruct(xs))
    return -sum(Base.Fix1(logpdf, transformed_dist), eachcol(obj.data))
end

# Some random data to estimate the density of.
xs = randn(2, 1000)

# Construct the objective.
f = NLLObjective(reconstruct_planarlayer, MvNormal(2, 1), xs)

println("Initial loss = $(f(xs_init)) at xs_init = $(xs_init)")
```

Now we can train the flow using gradient descent:

```@example normalizing-flows
using ForwardDiff: ForwardDiff

function train(xs_init, niters; stepsize=1e-3)
    xs = xs_init
    for i in 1:niters
        grad = ForwardDiff.gradient(f, xs)
        @. xs = xs - (stepsize * grad)
    end
    return xs
end
xs_trained = train(xs_init, 1000)

println("Final loss = $(f(xs_trained)) at xs_trained = $(xs_trained)")
```

Finally, we can sample from the trained flow and check that the samples have approximately zero mean and identity covariance (as expected given that our data was sampled using `randn`):

```@example normalizing-flows
samples = rand(transformed(f.basedist, f.reconstruct(xs_trained)), 1000);

# mean ≈ [0, 0], cov ≈ I
mean(eachcol(samples)), cov(samples; dims=2)
```

More complex flows can be created by composing multiple layers, e.g. `PlanarLayer(10) ∘ PlanarLayer(10) ∘ RadialLayer(10)`.
