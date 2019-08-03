using Zygote, LinearAlgebra
using PyCall
# Make sure pytorch is installed

torch = pyimport("torch")

# From Mike Innes' JuliaCon 2019 talk
# https://youtu.be/OcUXjk7DFvU?t=1106
Zygote.@adjoint function pycall(f, x...; kw...)
    x = map(py, x)
    y = pycall(f, x...; kw...)
    y.detach().numpy(), function (ȳ)
        y.backward(gradient = py(ȳ))
        (nothing, map(x->x.grad.numpy(), x)...)
    end
end

# Non-symmetric version, which gives complex eigenvalues
# function torch_eig(A)
#     tA = torch.from_numpy(A)
#     out = torch.eig(tA)
#     evs = out[1].numpy()
#     return [ evs[j, 1] + im*evs[j,2] for j = 1:size(evs,1) ]
# end

function torch_symeig(A)
    tA = torch.from_numpy(A)
    out = torch.symeig(tA)
    evs = out[1].numpy()
    return evs
end

function test_eigmin(A)
    evals = torch_symeig(A)
    return minimum(evals)
end


A = rand(4, 4); A = A + A';
@show eigmin(A)
@show test_eigmin(A)
@show Zygote.gradient(test_eigmin, A)
