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


function MagnetBatch(model, x, y, k)
    # idealy y should have the same number of elements from all classes
    pw_matrix = pairwise(model, x, x)
    pw_labels = pairwise(==, y, y)
    diag_ones = diagm(ones(length(y)))
    max_ = maximum(pw_matrix)

    # k closest (positive)
    
    same_class_pdm = pw_matrix .* (pw_labels .- diag_ones)
    [same_class_pdm.==0] .+= max_ + 1
    sorted_idx = sortperm(same_class_pdm, dims=2) # sorted columns / row-wise
    k_pos_closest = same_class_pdm[sorted_idx][:,1:k] # keep k columns
    

end


# Batching function
function TripletCreation(type_, data, batch_size, metric)
    if type_=="balanced"
        return SampleTriplets(data[1], data[2], batch_size, true)
    elseif type_=="batch_hard"
        batch_ = randperm(length(data[2]))[1:batch_size]
        return OfflineBatchHardTriplets(metric, data[1][batch_], data[2][batch_])
    elseif type_=="switching"
        if rand() < 0.5
            TripletCreation("balanced", data, batch_size, metric)
        else
            TripletCreation("batch_hard", data, batch_size, metric)
        end
    else
        @error "unknown triplet creation"
    end
end