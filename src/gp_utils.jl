using AbstractGPs, GPLikelihoods, Distributions, KernelFunctions, ApproximateGPs
#using Mill, HMilDistance

MillType = AbstractArray{<:Mill.AbstractMillNode}
MillVec = AbstractVector{<:Mill.AbstractMillNode}

abstract type AbstractHMillKernel <: KernelFunctions.Kernel end

# Tailor-made functions for GP libraries to accept HMill data

reduce_catobs(x) = Zygote.ignore(()->HMillDistance.preprocess_empty_bags(reduce(catobs, x))) # catobs does not have rrule so it needs to be ignored by AD
KernelFunctions.kernelmatrix(κ::AbstractHMillKernel, x::MillVec) = κ(reduce_catobs(x))

# This fixes error which comes from missing / unfitting chain rule for diagnoal matrices
#+(a::Diagonal, b::@NamedTuple{diag::Vector{Float64}}) = Diagonal(a.diag + b.diag)

function AbstractGPs.FiniteGP(f::AbstractGPs.AbstractGP, x::MillType, σ²::Real=default_σ²)
    return FiniteGP(f, x, Fill(σ², length(x)))
end


# just simple helper function
function KernelSelector(name::String; trainable::Bool=false)
    if name == "Laplacian" 
        return dist -> LaplacianHMillKernel(dist, γ=1.0, trainable=trainable)
    elseif name == "Gaussian"
        return dist -> GaussianHMillKernel(dist, γ=2.0, trainable=trainable)
    elseif name == "Matern32"
        return dist -> Matern32HMillKernel(dist)
    else 
        @error "Unknown kernel option"
    end
end



# Kernels
struct LaplacianHMillKernel <:AbstractHMillKernel 
    γ::Union{Vector, Number} # if γ<: Number -> nontrainable param ; if γ <: Vector -> trainable param #TODO think of better way
    d::AbstractMetric
end

"""
Constructor for LaplacianHMillKernel

    k(x,x') = exp( -γ ̇d(x,x'))

    d::AbstractMetric   ... base matric function
    γ::Real             ... scale parameter
    trainabel::Bool     ... bool if to optimise γ or not
"""
LaplacianHMillKernel(d::AbstractMetric; γ::Real=1.0, trainable::Bool=false, kwargs...) = LaplacianHMillKernel(trainable ? [γ] : γ, d)


Flux.@functor LaplacianHMillKernel
Flux.trainable(k::LaplacianHMillKernel) = (k.d, k.γ) # we can do if/else "trainable" here but Flux.destructure will ignore this anyway

(k::LaplacianHMillKernel)(x::AbstractMillNode,y::AbstractMillNode) = exp.(-only(k.γ) .* mean(k.d(x,y)))
(k::LaplacianHMillKernel)(x::Mill.AbstractMillNode) = exp.(-only(k.γ) .* k.d(x,x))

# Gaussian kernel
struct GaussianHMillKernel <:AbstractHMillKernel 
    γ::Union{Vector, Number} # if γ<: Number -> nontrainable param ; if γ <: Vector -> trainable param #TODO think of better way
    d::AbstractMetric
end

"""
Constructor for GaussianHMillKernel

    k(x,x') = exp( - d(x,x')^2 / γ)

    d::AbstractMetric   ... base matric function
    γ::Real             ... scale parameter
    trainabel::Bool     ... bool if to optimise γ or not
"""
GaussianHMillKernel(d::AbstractMetric; γ::Real=2.0, trainable::Bool=false, kwargs...) = GaussianHMillKernel(trainable ? [γ] : γ, d)


Flux.@functor GaussianHMillKernel
Flux.trainable(k::GaussianHMillKernel) = (k.d, k.γ) # we can do if/else "trainable" here but Flux.destructure will ignore this anyway

(k::GaussianHMillKernel)(x::AbstractMillNode,y::AbstractMillNode) = exp.(-(mean(k.d(x,y)) .^2) ./ only(k.γ))
(k::GaussianHMillKernel)(x::Mill.AbstractMillNode) = exp.(-(k.d(x,x) .^2) ./ only(k.γ))


# Matérn 32 kernel 
struct Matern32HMillKernel <:AbstractHMillKernel #<: KernelFunctions.Kernel
    d::AbstractMetric
end
#Matern32HMillKernel(d::AbstractMetric; kwargs...) = Matern32HMillKernel(d)

Flux.@functor Matern32HMillKernel
Flux.trainable(k::Matern32HMillKernel) = (k.d,)

function (k::Matern32HMillKernel)(x::AbstractMillNode, y::AbstractMillNode)
    dist_ = mean(k.d(x,y))
    return (1.0 .+ (sqrt(3) .* dist_)) .* exp.(-sqrt(3) .* dist_)
end

function (k::Matern32HMillKernel)(x::AbstractMillNode)
    dist_ = k.d(x,y)
    return (1.0 .+ (sqrt(3) .* dist_)) .* exp.(-sqrt(3) .* dist_)
end

