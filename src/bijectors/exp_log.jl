#############
# Exp & Log #
#############

struct Exp <: Bijector end
struct Log <: Bijector end

inv(b::Exp) = Log()
inv(b::Log) = Exp()

transform(b::Exp, y) = exp.(y)
transform(b::Log, x) = log.(x)

logabsdetjac(b::Exp, x) = sum(x)
logabsdetjac(b::Log, x) = -sum(log, x)

function forward(b::Log, x)
    y = transform(b, x)
    return (result = y, logabsdetjac = -sum(y))
end
