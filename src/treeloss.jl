# ProductNode 
Lₚ(p,w) = dropdims(sqrt.(sum(w .* (p.^2), dims=3)), dims=3) 
# BagNode loss 
Lᵦ(x) = CD(x, Flux.mean) 

# Chamfer Distance on Pairwise Distance Matrix
CD(pm, agg=mean) = agg(minimum(pm, dims=1)) + agg(minimum(pm, dims=2))
NormLevenshtein(x,y) = Levenshtein()(x,y)/min(length(x), length(y))

Lₐ(x::ArrayNode{<:NGramMatrix{String}}, y::ArrayNode{<:NGramMatrix{String}}) = pairwise(NormLevenshtein, x.data.S, y.data.S)

Lₐ(x::ArrayNode{<:Flux.OneHotArray}, y::ArrayNode{<:Flux.OneHotArray}) = Lₐ(x.data, y.data)
Lₐ(x::Flux.OneHotArray, y::Flux.OneHotArray) = sqrt.(Distances.pairwise(Distances.SqEuclidean(), x, y))

Lₐ(x::ArrayNode{<:Array}, y::ArrayNode{<:Array}) = Lₐ(x.data, y.data)
Lₐ(x::ArrayNode{<:Array}, y::Array) = Lₐ(x.data, y)
Lₐ(x::Array, y::ArrayNode{<:Array}) = Lₐ(x, y.data)
Lₐ(x::Array, y::Array) = sqrt.(Distances.pairwise(Distances.SqEuclidean(), x, y))

function Lₐ(x::Array{<:Union{Missing, Number}}, y::Array{<:Union{Missing, Number}})
    x̂ = copy(x); x̂[ismissing.(x̂)] .= 0
    ŷ = copy(y); ŷ[ismissing.(ŷ)] .= 0
    sqrt.(Distances.pairwise(Distances.SqEuclidean(), x̂, ŷ))
end


Lₐ(x::ArrayNode{<:MaybeHotMatrix}, y::ArrayNode{<:MaybeHotMatrix}) = Lₐ(x.data, y.data)

function Lₐ(x::MaybeHotMatrix, y::MaybeHotMatrix) 
    xsize, ysize = size(x), size(y);
    x̂ = copy(x); 
    ŷ = copy(y); 
    x̂[:,ismissing.(x[1,:])] .= onehot(xsize[1], 1:xsize[1]); # using OneHotArrays
    ŷ[:,ismissing.(y[1,:])] .= onehot(ysize[1], 1:ysize[1]); # using OneHotArrays

    return Lₐ(x̂, ŷ)    
end

"""
function zerocardinality(bag)
    if isempty(bag)
        return true
    elseif length(bag) == 1 && length(bag[1]) == 0
        return true
    else
        return false
    end
end
"""



function weighted_tree_distance(kb_true::KnowledgeBase, kb_pred::KnowledgeBase, weights::Union{Dict, WeightStruct}, node::Tuple, key="*"; just_in::Bool=true)
    if node[1] == "ArrayNode"
        loss_ = Lₐ(kb_true["$(key)_in"], kb_pred["$(key)_in"])
    elseif node[1] == "ProductNode"
        children = collect(keys(node[2]))
        elements = map(key-> weighted_tree_distance(kb_true, kb_pred, weights, node[2][key], String(key); just_in=just_in), children)# weights[key] *
        product = cat(elements..., dims=3)
        weights_ = reshape(map(key->weights[key], children), 1,1,:) # the same dimension as product
        loss_ = Lₚ(product, weights_) 
    elseif node[1] == "BagNode"
        children = collect(keys(node[2])) 
        if zerocardinality(kb_true["$(key)_bags"]) & zerocardinality(kb_pred["$(key)_bags"])
            #@error "zerocardianlity 0&0"
            loss_ = 0
        elseif zerocardinality(kb_true["$(key)_bags"])
            #@error "zerocardianlity 0&1"
            elements = map(key->weighted_tree_distance(kb_pred, weights, node[2][key], String(key); just_in=just_in), children)
            out_bag = kb_pred["$(key)_bags"]
            loss_ = GHMill.block_segmented_norm(elements[1], 1:1, out_bag, Lᵦ) 
        elseif zerocardinality(kb_pred["$(key)_bags"])
            #@error "zerocardianlity 1&0"
            elements = map(key->weighted_tree_distance(kb_true, weights, node[2][key], String(key); just_in=just_in), children)
            in_bag = kb_true["$(key)_bags"] 
            loss_ = GHMill.block_segmented_norm(elements[1]', in_bag, 1:1, Lᵦ)
        else
            elements = map(key->weighted_tree_distance(kb_true, kb_pred, weights, node[2][key], String(key); just_in=just_in), children)
            in_bag = kb_true["$(key)_bags"]
            out_bag = kb_pred["$(key)_bags"]
            loss_ = GHMill.block_segmented_norm(elements[1], in_bag, out_bag, Lᵦ)
        end
    else
        error("unknown node type")
    end
    return loss_
end



function weighted_tree_distance(kb::KnowledgeBase, weights::Union{Dict, WeightStruct}, node::Tuple, key="*"; just_in::Bool=true)
    if node[1] == "ArrayNode"
        #kb_pred_ = (just_in) ? kb_pred["$(key)_in"].data : kb_pred["$(key)_out"]
        loss_ = Lₐₐ(kb["$(key)_in"])
    elseif node[1] == "ProductNode"
        children = collect(keys(node[2]))
        elements = map(key-> weighted_tree_distance(kb, weights, node[2][key], String(key); just_in=just_in), children)# weights[key] *
        product = cat(elements..., dims=3)
        weights_ = reshape(map(key->weights[key], children), 1,1,:) # the same dimension as product
        loss_ = Lₚ(product, weights_) 
    elseif node[1] == "BagNode"
        children = collect(keys(node[2])) 
        if zerocardinality(kb["$(key)_bags"])
            loss_ = 0
        else
            elements = map(key->weighted_tree_distance(kb, weights, node[2][key], String(key); just_in=just_in), children)
            in_bag = kb["$(key)_bags"]
            loss_ = GHMill.block_segmented_norm(elements[1], 1:1, in_bag, Lᵦ)# TODO fixme:
        end
    else
        error("unknown node type")
    end
    return loss_
end

Lₐₐ(x::ArrayNode{<:Flux.OneHotArray}) = Lₐₐ(x.data)
Lₐₐ(x::Flux.OneHotArray) = sqrt.(sum(x .^2, dims=1))

Lₐₐ(x::ArrayNode{<:Array}) = Lₐₐ(x.data)
Lₐₐ(x::Array) = sqrt.(sum(x .^2, dims=1))

Lₐₐ(x::ArrayNode{<:NGramMatrix{String}}) = ones(Float32, 1, size(x.data)[2])
Lₐₐ(x::ArrayNode{<:NGramMatrix{Union{String,Missing}}}) = ones(Float32, 1, size(x.data)[2])

#Lₐₐ(x::ArrayNode{<:MaybeHotMatrix}) = Lₐₐ(x.data, y.data)

function Lₐₐ(x::ArrayNode{<:MaybeHotMatrix}) 
    xsize = size(x.data);
    x̂ = copy(x.data);   
    x̂[:,ismissing.(x.data[1,:])] .= onehot(xsize[1], 1:xsize[1]); # using OneHotArrays

    return Lₐₐ(x̂)    
end

# julia> val[1][2][:interactions].data[:records].bags|> isempty
# true