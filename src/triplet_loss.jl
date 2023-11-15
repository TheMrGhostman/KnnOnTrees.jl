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

#triplet_loss(model, xₐ, xₚ, xₙ, α=0) = max(mean(model.(xₐ, xₚ) .- model.(xₐ, xₙ) .+ α), 0)


function OfflineBatchHardTriplets(model, x, y)
    # special case for us
    pw_matrix = pairwise(model, x, x)
    pw_labels = pairwise(==, y, y)
    diag_ones = diagm(ones(length(y)))

    #originaly you should backpropagate this all, but we have special case
    _, argmax_hp = findmax(pw_matrix .* (pw_labels .- diag_ones), dims=2) # for symetric matrix is dim=1 and dim=2 the same
    #valid_negatives = pw_matrix .* (1 .- pw_labels)
    max_ = maximum(pw_matrix)
    #_, argmax_hn = findmin(pw_matrix .+ (max_ .* (1 .-(pw_labels .- diag_ones))), dims=2) 
    _, argmax_hn = findmin(pw_matrix .+ (max_ .* pw_labels), dims=2) 
    #return argmax_hp, argmax_hn
    argmax_hp = map(o->o[2], argmax_hp)
    argmax_hn = map(o->o[2], argmax_hn)
    xₐ, xₚ, xₙ = x, x[argmax_hp], x[argmax_hn] 
    return xₐ, xₚ, xₙ
end

