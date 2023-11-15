using MLUtils, Mill, Flux, Distances

abstract type AbstractMetric end

struct LeafMetric <: AbstractMetric
    metric
    type
    keyname
end

LeafMetric(metric) = LeafMetric(metric, "Any", "0")

Flux.@functor LeafMetric
#Flux.trainable(ws::LeafMetric) = () #TODO check this

function (m::LeafMetric)(x::ArrayNode, y::ArrayNode)
    if isempty(x.data) && isempty(y.data)
        0
    elseif isempty(y.data)
        m.metric(x.data, zeros_like(x.data))
    elseif isempty(x.data)
        m.metric(zeros_like(y.data), y.data)
    else
        m.metric(x.data, y.data)
    end
end

#(m::LeafMetric)(x::ArrayNode, y::ArrayNode) = m.metric(x.data, y.data)

struct ProductMetric{T, U, V} <: AbstractMetric
    ms::T # instance metrices
    pm::U # product metric
    weights::V #TODO think about this
    keyname
end

Flux.@functor ProductMetric
#Flux.trainable(m::ProductMetric) = (m.weights) # TODO check this

function (m::ProductMetric{<:NamedTuple{M}})(x::ProductNode{<:NamedTuple{X}}, y::ProductNode{<:NamedTuple{Y}}) where {M,X,Y} 
    @assert issubset(M, X) && issubset(M, Y)
    elements = map(M) do k
        m.ms[k](x.data[k], y.data[k])
    end
    product = cat(elements..., dims=3)
    weights_ = reshape(collect(map(k->m.weights[k], M)), 1,1,:)
    m.pm(product, weights_)
end


struct SetMetric <: AbstractMetric
    im # instance metric
    sm # set metric
    keyname
end

Flux.@functor SetMetric
#Flux.trainable(m::SetMetric) = ()  # TODO check this

function (m::SetMetric)(x::BagNode, y::BagNode) 
    if all(zerocardinality.([x.bags, y.bags]))  return zeros(Float32, 1,1) end
    elements = m.im(x.data, y.data)
    xbags = zerocardinality(x.bags) ? range(1,1) : x.bags
    ybags = zerocardinality(y.bags) ? range(1,1) : y.bags
    block_segmented_norm(elements, xbags, ybags, m.sm)
end


function reflectmetric(
    x; 
    prod_metric=WeightedProductMetric, 
    set_metric=ChamferDistance, 
    leaf_metrics=Dict(
        :continuous => Pairwise_Euclidean,
        :categorical => Pairwise_Cityblock,
        :string => Pairwise_Levenstein
    ), 
    weight_sampler=ones, 
    weight_transform=identity,
    s="0"
)
    return _reflectmetric(x, prod_metric, set_metric, leaf_metrics, weight_sampler, weight_transform, s)
end

# TODO add maybe oh / missing / 

function _reflectmetric(x::ArrayNode{<:Array}, prod_metric, set_metric, leaf_matric, weight_sampler, weight_transform, s)
    return LeafMetric(leaf_matric[:continuous], "con", Mill.stringify(s))
end

function _reflectmetric(x::ArrayNode{<:Flux.OneHotArray}, prod_metric, set_metric, leaf_matric, weight_sampler, weight_transform, s) 
    return LeafMetric(leaf_matric[:categorical], "cat", Mill.stringify(s))
end

function _reflectmetric(x::ArrayNode{<:NGramMatrix}, prod_metric, set_metric, leaf_matric, weight_sampler, weight_transform, s) 
    return LeafMetric(leaf_matric[:string], "str", Mill.stringify(s))
end


function _reflectmetric(x::AbstractProductNode, prod_metric, set_metric, leaf_matric, weight_sampler, weight_transform, s)
    c = Mill.stringify(s)
    n = length(x.data)
    ks = keys(x.data)
    ms = [_reflectmetric(x.data[k], prod_metric, set_metric, leaf_matric, weight_sampler, weight_transform, s * Mill.encode(i, n))
                  for (i, k) in enumerate(ks)]
    weights = WeightStruct(Mill._remap(x.data, weight_sampler(length(ms))), weight_transform)
    ms = Mill._remap(x.data, ms)
    return ProductMetric(ms, prod_metric, weights, c)
end

function _reflectmetric(x::AbstractBagNode, prod_metric, set_metric, leaf_matric, weight_sampler, weight_transform, s)
    c = Mill.stringify(s)
    im = _reflectmetric(x.data, prod_metric, set_metric, leaf_matric, weight_sampler, weight_transform, s * Mill.encode(1, 1))
    return SetMetric(im, set_metric, c)
end



"""
dist_(x::Array, y::Array) = sqrt.(Distances.pairwise(Distances.SqEuclidean(), x, y))
dist_(x::Flux.OneHotArray, y::Flux.OneHotArray) = Distances.pairwise(Distances.Cityblock(), x, y) ./2 
dist_(x::ArrayNode{<:NGramMatrix{String}}, y::ArrayNode{<:NGramMatrix{String}}) = pairwise(NormLevenshtein, x.data.S, y.data.S)
"""

NormLevenshtein(x,y) = Levenshtein()(x,y)/min(length(x), length(y))
Pairwise_Levenstein(x::ArrayNode{<:NGramMatrix{String}}, y::ArrayNode{<:NGramMatrix{String}}) = pairwise(NormLevenshtein, x.data.S, y.data.S)


Pairwise_Euclidean(x::Array, y::Array) = sqrt.(Distances.pairwise(Distances.SqEuclidean(), x, y) .+ 1f-20)


Pairwise_Cityblock(x::Flux.OneHotArray, y::Flux.OneHotArray) = Distances.pairwise(Distances.SqEuclidean(), x, y) ./2 # equivalent to Cityblock distance
#Distances.pairwise(Distances.Cityblock(), x, y) ./2 


WeightedProductMetric(p,w) = dropdims(sqrt.(sum(w .* (p.^2), dims=3) .+ 1f-20), dims=3)
ChamferDistance(pm, agg=mean) = agg(minimum(pm, dims=1)) + agg(minimum(pm, dims=2))


function block_segmented_norm(x, seg1, seg2, norm::Function=sum)
    [norm(x[sᵢ, sⱼ]) for sᵢ in seg1, sⱼ in seg2]
end


