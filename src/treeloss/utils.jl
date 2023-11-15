
"""
# Differentable Dict-like struct

W = Dict{Any, Any}("Kg" => 1.0f0, "2" => 1.0f0, "6" => 1.0f0, "KE" => 1.0f0, "L*" => 1.0f0, "Kd" => 1.0f0, "A" => 1.0f0, "Ke" => 1.0f0, "E" => 1.0f0, "Kf" => 1.0f0, "I" => 1.0f0, "KU" => 1.0f0, "Kk" => 1.0f0)
ws = WeightStruct(W)

grad = gradient(
    ()->begin
        sum(map((k,y)->ws[k]*y, ws.keys, 1:13))
    end,
    Flux.params(ws)     
) 
"""

struct WeightStruct
    keys
    values
    transform::Function # Transformation of weights 
end

WeightStruct(x::Dict, transfrom::Function=identity) = WeightStruct(collect(keys(x)), Float32.(collect(values(x))), transform)
WeightStruct(x::NamedTuple, transform::Function=identity) = WeightStruct(collect(keys(x)), Float32.(collect(values(x))), transform)

Flux.@functor WeightStruct

Flux.trainable(ws::WeightStruct) = (values=ws.values,)

function Base.getindex(ws::WeightStruct, k::Union{String, Symbol})
    idx = Zygote.@ignore(findmax(ws.keys.==k)[2])
    return ws.transform.(ws.values[idx])
end


function _sample_triplet_indexes(y, balanced::Bool=true)
    anchors = (balanced) ? findall(y .== sample(unique(y))) : range(1,length(y));
    anchor_idx = sample(anchors)

    positives = filter(x->x != anchor_idx, findall(y .== y[anchor_idx]))
    negatives = findall(y .!= y[anchor_idx])
    @assert length(anchors) != 0 || length(positives) != 0 || length(negatives) != 0
    return anchor_idx, sample(positives), sample(negatives)
end

function SampleTriplets(X, y, batchsize=10, balanced::Bool=true)
    index_matrix = hcat(collect.([_sample_triplet_indexes(y, balanced) for _=1:batchsize])...);
    iₐ, iₚ, iₙ = index_matrix[1,:], index_matrix[2,:], index_matrix[3,:]
    return X[iₐ], X[iₚ], X[iₙ]
end


function zerocardinality(bag)
    if isempty(bag)
        return true
    elseif length(bag) == 1 && length(bag[1]) == 0
        return true
    else
        return false
    end
end

