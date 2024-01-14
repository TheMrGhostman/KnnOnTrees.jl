using AbstractGPs, GPLikelihoods, Distributions, KernelFunctions, ApproximateGPs
#using Mill, HMillDistance

MillType = AbstractArray{<:Mill.AbstractMillNode}
MillVec = AbstractVector{<:Mill.AbstractMillNode}

abstract type AbstractHMILKernel <: KernelFunctions.Kernel end

# Tailor-made functions for GP libraries to accept HMill data

reduce_catobs(x) = Zygote.ignore(()->reduce(catobs, x)) # catobs does not have rrule so it needs to be ignored by AD
KernelFunctions.kernelmatrix(κ::AbstractHMILKernel, x::MillVec) = κ(reduce_catobs(x))

# This fixes error which comes from missing / unfitting chain rule for diagnoal matrices
#+(a::Diagonal, b::@NamedTuple{diag::Vector{Float64}}) = Diagonal(a.diag + b.diag)

function AbstractGPs.FiniteGP(f::AbstractGPs.AbstractGP, x::MillType, σ²::Real=default_σ²)
    return FiniteGP(f, x, Fill(σ², length(x)))
end


# Kernels
"""
LinearHMILKernel is equivalent to Linear Kernel with HMILDicence instead 
"""
struct LinearHMILKernel <:AbstractHMILKernel #<: KernelFunctions.Kernel
    d::AbstractMetric
end

Flux.@functor LinearHMILKernel
Flux.trainable(k::LinearHMILKernel) = (k.d,)

(k::LinearHMILKernel)(x::AbstractMillNode,y::AbstractMillNode) = mean(k.d(x,y)) # matrix/vector -> number
(k::LinearHMILKernel)(x::Mill.AbstractMillNode) = k.d(x,x) # special version


struct LaplacianHMILKernel <:AbstractHMILKernel #<: KernelFunctions.Kernel
    d::AbstractMetric
end
#LaplacianHMILKernel(γ::Real, d::AbstractMetric) = LaplacianHMILKernel([γ], d)

Flux.@functor LaplacianHMILKernel
Flux.trainable(k::LaplacianHMILKernel) = (k.d, )

(k::LaplacianHMILKernel)(x::AbstractMillNode,y::AbstractMillNode) = exp.(-mean(k.d(x,y)))
(k::LaplacianHMILKernel)(x::Mill.AbstractMillNode) = exp.(-k.d(x,x))


struct PolynomialHMILKernel <:AbstractHMILKernel #<: KernelFunctions.Kernel
    γ::Vector
    d::AbstractMetric
end
PolynomialHMILKernel(γ::Real, d::AbstractMetric) = PolynomialHMILKernel([γ], d)

Flux.@functor PolynomialHMILKernel
Flux.trainable(k::PolynomialHMILKernel) = (k.d, k.γ)

(k::PolynomialHMILKernel)(x::AbstractMillNode,y::AbstractMillNode) = (mean(k.d(x,y)) .+ 1) .^ k.γ[1]
(k::PolynomialHMILKernel)(x::Mill.AbstractMillNode) = (k.d(x,x) .+ 1) .^ k.γ[1]


struct Matern32HMILKernel <:AbstractHMILKernel #<: KernelFunctions.Kernel
    d::AbstractMetric
end

Flux.@functor Matern32HMILKernel
Flux.trainable(k::Matern32HMILKernel) = (k.d,)

function (k::Matern32HMILKernel)(x::AbstractMillNode, y::AbstractMillNode)
    dist_ = mean(k.d(x,y))
    return (1.0 .+ (sqrt(3) .* dist_)) .* exp.(-sqrt(3) .* dist_)
end

function (k::Matern32HMILKernel)(x::AbstractMillNode)
    dist_ = k.d(x,y)
    return (1.0 .+ (sqrt(3) .* dist_)) .* exp.(-sqrt(3) .* dist_)
end