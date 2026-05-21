# Defining a bijector

This page describes the minimum expected interface to implement a bijector.

In general, there are two pieces of information needed to define a bijector:

 1. The transformation itself, i.e., the map $b: \mathbb{R}^d \to \mathbb{R}^d$.

 2. The log-absolute determinant of the Jacobian of that transformation.
    For a transformation $b: \mathbb{R}^d \to \mathbb{R}^d$, the Jacobian at point $x \in \mathbb{R}^d$ is defined as:
    
    $$J_{b}(x) = \begin{bmatrix}
    \partial y_1/\partial x_1 & \partial y_1/\partial x_2 & \cdots & \partial y_1/\partial x_d \\
    \partial y_2/\partial x_1 & \partial y_2/\partial x_2 & \cdots & \partial y_2/\partial x_d \\
    \vdots & \vdots & \ddots & \vdots \\
    \partial y_d/\partial x_1 & \partial y_d/\partial x_2 & \cdots & \partial y_d/\partial x_d
    \end{bmatrix}$$
    
    where $y = b(x)$.

## The transform itself

The most efficient way to implement a bijector is to provide an implementation of:

```@docs; canonical=false
Bijectors.with_logabsdet_jacobian
```

!!! note
    
    `with_logabsdet_jacobian` is re-exported from ChangesOfVariables.jl, so if you want to avoid importing Bijectors.jl, you can implement `ChangesOfVariables.with_logabsdet_jacobian` instead.

If you define `with_logabsdet_jacobian(b, x)`, then you will automatically get default implementations of both `transform(b, x)` and `logabsdetjac(b, x)`, which respectively return the first and second value of that tuple.
So, in fact, you can implement a bijector by defining only `with_logabsdet_jacobian`.

If you prefer, you can implement `transform` and `logabsdetjac` separately, as described below.
Having manual implementations of these may also be useful if you expect either to be used heavily without the other.

### Transformation

```@docs; canonical=false
transform
```

If `transform(b, x)` is defined, then you will automatically get a default implementation of `b(x)` which calls that.

### Log-absolute determinant of the Jacobian

```@docs; canonical=false
Bijectors.logabsdetjac
```

## Inverse

Often you will want to define an inverse bijector as well.
To do so, you will have to implement:

```@docs; canonical=false
Bijectors.inverse
```

!!! note
    
    `inverse` is re-exported from InverseFunctions.jl, so the same note as for `with_logabsdet_jacobian` applies.

If `b` is a bijector, then `inverse(b)` should return the inverse bijector $b^{-1}$.

If your bijector subtypes `Bijectors.Bijector`, then you will get a default implementation of `inverse` which constructs `Bijectors.Inverse(b)`.
This may be easier than creating a second type for the inverse bijector.
Note that you will also need to implement the methods for `with_logabsdet_jacobian` (and/or `transform` and `logabsdetjac`) for the inverse bijector type.

If your bijector is not invertible, you can specify this here:

```@docs; canonical=false
Bijectors.isinvertible
```

## Distributions

If your bijector is intended for use with a distribution, i.e., it transforms random variables drawn from that distribution to Euclidean space, then you should also implement:

```@docs; canonical=false
Bijectors.bijector
```

which should return your bijector.

On top of that, you should also implement a method for `Bijectors.output_size(b, dist::Distribution)`:

```@docs; canonical=false
Bijectors.output_size
```

## Closed-form

If your bijector does _not_ have a closed-form expression (e.g. if it uses an iterative procedure), then this should be set to false:

```@docs; canonical=false
Bijectors.isclosedform
```

The default is `true` so you only need to set this if your bijector is not closed-form.
