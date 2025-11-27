# Bijectors.jl

This package implements functionality for transforming random variables to Euclidean space (and back).

For example, consider a random variable $X \sim \mathrm{Beta}(2, 2)$, which has support on $(0, 1)$:

```@example main
using Bijectors

x = rand(Beta(2, 2))
```

In this case, the [logit function](https://en.wikipedia.org/wiki/Logit) is used as the transformation:

$Y = \mathrm{logit}(X) = \log\left(\frac{X}{1 - X}\right).$

We can construct this function

```@example main
b = bijector(Beta(2, 2))
```

and apply it to `x`:

```@example main
y = b(x)
```

You can also obtain the log absolute determinant of the Jacobian of the transformation:

```@example main
y, ladj = with_logabsdet_jacobian(b, x)
```
