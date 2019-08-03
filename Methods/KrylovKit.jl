using Zygote, LinearAlgebra
using KrylovKit: eigsolve

test_eigmin(A::AbstractMatrix) = eigsolve(A, 1, :SR)[1][1]

A = rand(4, 4); A = A + A';

@show eigmin(A)
@show test_eigmin(A)
@show Zygote.gradient(test_eigmin, A)
