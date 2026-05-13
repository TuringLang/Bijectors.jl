using DifferentiationInterface
using FiniteDifferences
using Test

const REF_BACKEND = AutoFiniteDifferences(; fdm=central_fdm(5, 1))

function test_ad(f, backend, x; rtol=1e-6, atol=1e-6)
    @info "testing AD for function $f with $backend"
    ref_gradient = DifferentiationInterface.gradient(f, REF_BACKEND, x)
    gradient = DifferentiationInterface.gradient(f, backend, x)
    @test isapprox(gradient, ref_gradient; rtol=rtol, atol=atol)
end
