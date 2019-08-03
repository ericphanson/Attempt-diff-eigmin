using Zygote, LinearAlgebra
using ArnoldiMethod: partialschur, SR

function test_eigmin(A::AbstractMatrix)
    decomp, history = partialschur(A; nev = 1, which = SR())
    ev = minimum(real.(decomp.eigenvalues))
    return ev
end

A = rand(4, 4); A = A + A';

@show eigmin(A)
@show test_eigmin(A)
@show Zygote.gradient(test_eigmin, A)
