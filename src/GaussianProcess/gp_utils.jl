using AbstractGPs, GPLikelihoods, Distributions, KernelFunctions, ApproximateGPs
#using Mill, HMilDistance

MillType = AbstractArray{<:Mill.AbstractMillNode}
MillVec = AbstractVector{<:Mill.AbstractMillNode}

# Tailor-made functions for GP libraries to accept HMill data
reduce_catobs(x) = Zygote.ignore(()->HMillDistance.preprocess_empty_bags(reduce(catobs, x); extend=true, top_pn=true)) # catobs does not have rrule so it needs to be ignored by AD
KernelFunctions.kernelmatrix(κ::AbstractHMillKernel, x::MillVec) = κ(reduce_catobs(x))

function KernelFunctions.kernelmatrix_diag(κ::AbstractHMillKernel, x::MillVec)
    x_new = HMillDistance.preprocess_empty_bags.(x; extend=true, top_pn=true)
    return map(κ, x_new, x_new)
end

# This fixes error which comes from missing / unfitting chain rule for diagnoal matrices
#+(a::Diagonal, b::@NamedTuple{diag::Vector{Float64}}) = Diagonal(a.diag + b.diag)

function AbstractGPs.FiniteGP(f::AbstractGPs.AbstractGP, x::MillType, σ²::Real=default_σ²)
    return FiniteGP(f, x, Fill(σ², length(x)))
end


function MO_argmax(y, out_dims)
    predicted_ = softmax(hcat(chunk(y, out_dims)...), dims=1)
    max_, argmax_ = findmax(predicted_, dims=1)
    argmax_ = getindex.(argmax_, 1)
    return max_[:], argmax_[:]
end