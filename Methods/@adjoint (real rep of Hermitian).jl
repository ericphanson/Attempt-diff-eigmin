using Zygote, LinearAlgebra
using KrylovKit: eigsolve

# helper for `herm`
function f(x, y, i, j)
    if i == j
        return x/2 + im*zero(y)
    elseif i > j
        return x + im * y
    elseif i < j
        return zero(x) + im*zero(y)
    end
end

# helper for invherm
function finv(z, i, j)
    if i <= j
        return real(z)
    elseif i > j
        return imag(z)
    end
end

# Create a complex `d` by `d` Hermitian matrix from a `d^2`-dimensional real vector `v`
function herm(v::AbstractVector)
    d = isqrt(length(v))
    @assert d^2 == length(v)
    m = reshape(v, d, d)
    m = [ f(m[i,j], m[j,i], i, j) for i = 1:d, j = 1:d]
    m = m + m'
    return m
end

# Recover the vector representation
function invherm(A::AbstractMatrix)
    d = size(A,1)
    m = [ finv(A[i,j], i, j) for i = 1:d for j = 1:d ]
    return vec(m)
end

# Find the minimal eigenvalue using KrylovKit (from a vector representation)
function eigmin_vec(v::AbstractVector)
    A = herm(v)
    eigsolve(A, 1, :SR)[1][1] |> real
end

# Give an adjoint for `Zygote`, which technically only works in the non-degenerate case
Zygote.@adjoint function eigmin_vec(v::AbstractVector)
    A = herm(v)
    eigs = eigsolve(A, 1, :SR)
    eval = eigs[1][1]
    evec = eigs[2][1]
    adj = (evec * evec') |> invherm
    return eval |> real, c->(c * adj,)
end

A = rand(4, 4); A = A + A';
v = invherm(A)

@show eigmin(A)
@show eigmin_vec(v)
@show Zygote.gradient(eigmin_vec, v)
