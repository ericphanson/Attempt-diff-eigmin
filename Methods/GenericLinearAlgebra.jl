using Zygote, LinearAlgebra
using GenericLinearAlgebra: _eigvals!

function test_eigmin(A::AbstractMatrix)
    vals = _eigvals!(copy(A))
    ev = minimum(real.(vals))
    return ev
end

A = rand(4, 4); A = A + A';

@show eigmin(A)
@show test_eigmin(A)
@show Zygote.gradient(test_eigmin, A)
