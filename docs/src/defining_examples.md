# Defining a bijector: examples

Here we provide two different worked examples of defining a custom bijector.

## Cyclic permutation

We start with something simple: a bijector that performs a cyclic permutation of the elements of a vector.

```@example cyclic
using Bijectors

struct CircShift <: Bijector
    shift::Int
end
```

As described in the [previous page](./defining.md), the only function you absolutely _must_ implement is `with_logabsdet_jacobian`.

Let's think for a moment about what the Jacobian is.
`CircShift` is a mapping from `ℝⁿ → ℝⁿ` that permutes the elements of the input vector.
For example, `CircShift(1)` would map a length-3 vector `x = [x1, x2, x3]` to `y = [x3, x1, x2]`.

That means that the Jacobian matrix is

$$J = \begin{bmatrix}
\partial y_1/\partial x_1 & \partial y_1/\partial x_2 & \partial y_1/\partial x_3 \\
\partial y_2/\partial x_1 & \partial y_2/\partial x _2 & \partial y_2/\partial x_3 \\
\partial y_3/\partial x_1 & \partial y_3/\partial x_2 & \partial y_3/\partial x_3
\end{bmatrix}
= \begin{bmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}.$$

In general, the Jacobian of such a transformation is a [permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix).
The determinant of a permutation matrix is either `1` or `-1`, depending on whether the permutation is even or odd.
(In this case, it is even; but it could be odd for other shifts and/or input sizes.)
This means that the log-absolute determinant of the Jacobian is always `0`.

We can now implement `with_logabsdet_jacobian`.

```@example cyclic
function Bijectors.with_logabsdet_jacobian(
    b::CircShift, x::AbstractVector{T}
) where {T<:Real}
    y = circshift(x, b.shift)
    return y, zero(T)
end
```

It is good practice to let the type of the input determine the type of the log-Jacobian term here.
However, you might also ask: since a cyclic permutation is also well-defined for arrays of non-real types, should we also allow that?
We can do so by creating a new method, but we would have to make a choice as to the type of the log-Jacobian term, since we cannot derive it from the input type.
Here, we will choose `Float64`:

```@example cyclic
import ChangesOfVariables: with_logabsdet_jacobian

function with_logabsdet_jacobian(b::CircShift, x::AbstractVector)
    y = circshift(x, b.shift)
    return y, 0.0
end
```

With this defined, we can now benefit from a host of automatic definitions:

```@example cyclic
b = CircShift(1)
x = [1.0, 2.0, 3.0]
b(x)
```

```@example cyclic
logabsdetjac(b, x)
```

We can also define the inverse bijector.
A default definition for `inverse(b)` already exists: it would return `Bijectors.Inverse(b)`.
But, if we used this default definition, we would have to also define `with_logabsdet_jacobian(::Inverse{CircShift}, y)`.
We can save ourselves this hassle by overloading the method:

```@example cyclic
import InverseFunctions: inverse

inverse(b::CircShift) = CircShift(-b.shift)
```

Now we can use the inverse bijector:

```@example cyclic
y = b(x)
inverse(b)(y) == x
```

!!! note
    
    Bijectors re-exports both `with_logabsdet_jacobian` as well as `inverse`, so you don't need to import them separately if Bijectors is already a dependency.
    Conversely, if you don't want to depend on Bijectors.jl directly, you can just import these functions from their respective packages.

## Stereographic projection

Now, we'll look at a more complex example: a stereographic projection mapping points on the unit sphere (i.e., length-3 vectors $x$ for which $x_1^2 + x_2^2 + x_3^2 = 1$), to points in the plane `ℝ²` (i.e., length-2 vectors $y$ whose elements are unconstrained).

The relevant formulae are given [here on Wikipedia](https://en.wikipedia.org/wiki/Stereographic_projection#First_formulation).
The forward transform (from sphere to plane) is:

$$y_1 = \frac{x_1}{1 - x_3}; \qquad y_2 = \frac{x_2}{1 - x_3}$$

```@example stereographic
using Bijectors

struct StereographicProj <: Bijector end
function (s::StereographicProj)(x::AbstractVector{T}) where {T<:Real}
    y = similar(x, 2)
    denom = one(T) - x[3]
    y[1] = x[1] / denom
    y[2] = x[2] / denom
    return y
end
```

!!! warning
    
    This will return `[Inf, Inf]` if `x[3] == 1` (the 'north pole' of the sphere), which may potentially make downstream computations fail. One potential way around this is to add `eps(T)` to the denominator to avoid it ever being zero: you will sometimes see this trick used in Bijectors.jl. However, be aware that the reverse transform has to also be modified accordingly so that the two transforms remain inverses of each other!

When it comes to computing the Jacobian, we find ourselves in a spot of bother.
The partial derivatives themselves can ostensibly be computed using fairly straightforward calculus:

$$J = \begin{bmatrix}
\partial y_1/\partial x_1 & \partial y_1/\partial x_2 & \partial y_1/\partial x_3 \\
\partial y_2/\partial x_1 & \partial y_2/\partial x_2 & \partial y_2/\partial x_3
\end{bmatrix}
= \begin{bmatrix}
1/(1 - x_3) & 0 & x_1/(1 - x_3)^2 \\
0 & 1/(1 - x_3) & x_2/(1 - x_3)^2
\end{bmatrix}$$

but since our mapping is from `ℝ³ → ℝ²`, the Jacobian matrix is not square, and so we cannot compute its determinant!

To fix this, we need to realise that $x_1$, $x_2$, and $x_3$ are not really independent at all.
The partial derivatives we computed above treated them as independent variables!
In reality, they must satisfy the constraint $x_1^2 + x_2^2 + x_3^2 = 1$, which means that

$$x_3 = \pm \sqrt{1 - x_1^2 - x_2^2},$$

and thus

$$\frac{\partial x_3}{\partial x_1} = -\frac{x_1}{x_3}; \qquad \frac{\partial x_3}{\partial x_2} = -\frac{x_2}{x_3}.$$

(Note that this is true regardless of which sign $x_3$ has.)

In effect, we are treating $x_3$ as a function of $x_1$ and $x_2$, rather than as an independent variable.
This means that we can construct a Jacobian using only $x_1$ and $x_2$ as inputs, and thus obtain a square Jacobian matrix.

For example, we can recompute the derivative of $y_1$ with respect to $x_1$, but this time also making sure to include the dependence of $x_3$ on $x_1$.

$$\begin{align*}
\frac{\partial y_1}{\partial x_1}
&= \frac{\partial}{\partial x_1} [x_1(1 - x_3)^{-1}] \\
&= (1 - x_3)^{-1} + x_1 (-1)(1 - x_3)^{-2} \left(\frac{\partial}{\partial x_1}(1 - x_3)\right) \\
&= (1 - x_3)^{-1} - x_1 (1 - x_3)^{-2} \left(\frac{x_1}{x_3}\right) \\
&= \frac{1}{1 - x_3} - \frac{x_1^2}{x_3 (1 - x_3)^2}
\end{align*}$$

A similar strategy for all the other partial derivatives gives us the Jacobian

$$J = \begin{bmatrix}
\dfrac{1}{1 - x_3} - \dfrac{x_1^2}{x_3 (1 - x_3)^2} & -\dfrac{x_1 x_2}{x_3 (1 - x_3)^2} \\
-\dfrac{x_1 x_2}{x_3 (1 - x_3)^2} & \dfrac{1}{1 - x_3} - \dfrac{x_2^2}{x_3 (1 - x_3)^2}
\end{bmatrix}.$$

!!! note
    
    When you see $x_3$ here, don't think 'the variable $x_3$': it's just shorthand for $\pm \sqrt{1 - x_1^2 - x_2^2}$. (And recall that these formulae hold for both choices of sign.)

Its determinant very nicely simplifies to

$$\det(J) = -\frac{1}{x_3 (1 - x_3)^2},$$

the _absolute_ determinant being

$$|\det(J)| = \frac{1}{|x_3| (1 - x_3)^2},$$

(`(1 - x_3)^2` is always non-negative, of course); and thus

```@example stereographic
function Bijectors.logabsdetjac(b::StereographicProj, x::AbstractVector{T}) where {T<:Real}
    return -log(abs(x[3])) - (2 * log(one(T) - x[3]))
end
```

Phew!

Let's take a moment and check that we did indeed do this correctly.
To verify that the implementation of `logabsdetjac` is indeed correct, we can compare it against a Jacobian obtained via *automatic differentiation*.

If we try to calculate a Jacobian for `StereographicProj()`, we will just get a 2x3 matrix, which is not what we want.
So, we need to take an extra step to map from the independent coordinates of $x$ (i.e., $x_1$ and $x_2$) to the full 3D coordinates, and _then_ to the plane:

```@example stereographic
sgn = 1

function full_transform(x12)
    x3 = sgn * sqrt(one(eltype(x12)) - sum(x12 .^ 2))
    x123 = vcat(x12, x3)
    return StereographicProj()(x123)
end

import DifferentiationInterface as DI
using FiniteDifferences, LinearAlgebra
x = [0.3, 0.4, sgn * sqrt(1 - 0.3^2 - 0.4^2)]

adtype = DI.AutoFiniteDifferences(; fdm=central_fdm(5, 1))
jac = DI.jacobian(full_transform, adtype, x[1:2])
logjac = logabsdet(jac)[1]
```

Hopefully this is approximately the same!

```@example stereographic
logabsdetjac(StereographicProj(), x)
```

You can also rerun the code blocks above with `sgn = -1` to verify that our `logabsdetjac` implementation does indeed behave correctly for both positive and negative values of $x_3$.

When writing unit tests for a new bijector, it is a good idea to include comparisons like this to verify that the Jacobian is computed correctly.
The strategy used above to get square Jacobians is quite generally applicable, and is used for testing the bijectors for (e.g.) simplices and Cholesky factors.

Returning to the Bijectors interface, because we have defined the forward transform as well as `logabsdetjac`, we can just use these to implement `with_logabsdet_jacobian`:

```@example stereographic
function Bijectors.with_logabsdet_jacobian(s::StereographicProj, x)
    return s(x), logabsdetjac(s, x)
end
```

Or if we wanted to be more efficient, we might notice that `one(T) - x[3]` is computed both in `s(x)` as well as in `logabsdetjac`.
So we could also write:

```@example stereographic
function Bijectors.with_logabsdet_jacobian(
    s::StereographicProj, x::AbstractVector{T}
) where {T<:Real}
    denom = one(T) - x[3]  # Shared computation
    y = similar(x, 2)
    y[1] = x[1] / denom
    y[2] = x[2] / denom
    logjac = -log(abs(x[3])) - (2 * log(denom))
    return y, logjac
end
```

Of course, this alone is unlikely to save any meaningful amount of time, but other bijectors may have more expensive computations that may be shared between both transform and log-Jacobian calculations.

The inverse bijector can be implemented in a very similar way (Wikipedia has the formulae as well), but is left as an exercise for the very willing reader!

Finally, suppose we had a distribution `UnitSphere`, where `rand(UnitSphere())` returned a random point on the unit sphere.
Something similar to this technically exists in Manifolds.jl, but we can also define a hacky version ourselves:

```@example stereographic
using Distributions

struct UnitSphere <: Distributions.ContinuousMultivariateDistribution end
Base.size(::UnitSphere) = (3,)
Base.rand(::UnitSphere) = normalize(rand(3))
```

Then, we could define

```@example stereographic
Bijectors.bijector(::UnitSphere) = StereographicProj()

# Not strictly needed for this example, but other usage may require it
Bijectors.output_size(::StereographicProj, ::UnitSphere) = (2,)
```

and that would allow us to construct, for example, transformed distributions:

```@example stereographic
td = transformed(UnitSphere())
rand(td)  # returns a random point in ℝ²
```

We didn't define `logpdf` for `UnitSphere`, but if we had, then we would also be able to make use of `logpdf(td, y)` and `Bijectors.logpdf_with_trans`.
