using Zygote, LinearAlgebra
using IterativeSolvers: lobpcg

function test_eigmin(A::AbstractMatrix)
    results = lobpcg(A, false, 1)
    ev = minimum(results.λ)
    return ev
end

A = rand(4, 4); A = A + A';

@show eigmin(A)
@show test_eigmin(A)
@show Zygote.gradient(test_eigmin, A)
