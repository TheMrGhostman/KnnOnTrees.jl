function knn(x_train, x_test, W::Dict, enc, structure; verbose::Bool=true)
    test_range = (verbose) ? tqdm(1:length(x_test)) : 1:length(x_test)
    train_range = 1:length(x_train)
    dist_matrix = zeros(length(x_train), length(x_test))
    for tst_idx in test_range
        kb_test, _ = enc(x_test[tst_idx])
        Threads.@threads for tr_idx in train_range
            kb_train, _ = enc(x_train[tr_idx])
            dist_ = weighted_tree_distance(kb_test, kb_train, W, structure, "*"; just_in = true) |> mean
            #weighted_treeloss
            dist_matrix[tr_idx, tst_idx] = dist_
        end
    end
    return dist_matrix
end

function knn(x_train, x_test, W::Dict; verbose::Bool=true)
    enc = reflectindecoder(x_train[1]; verbose=false)
    structure = identify_w(enc, Dict(), W; verbose=false)["*"] #TODO fixme W....
    knn(x_train, x_test, W, enc, structure; verbose=verbose)
end

function knn_tm(x_train, x_test, metrics::Function; verbose::Bool=false)
    test_range = (verbose) ? tqdm(1:length(x_test)) : 1:length(x_test)
    train_range = 1:length(x_train)
    dist_matrix = zeros(length(x_train), length(x_test))
    for tst_idx in test_range
        for tr_idx in train_range
            dist_ = metrics(x_test[tst_idx], x_train[tr_idx])
            dist_matrix[tr_idx, tst_idx] = dist_
        end
    end
    return dist_matrix
end

function knn_probs(pdm::Matrix, labels::Vector, k::Int) # not optimal function in any way!!!
    # pdm ~ n x m, where n ~ num samples in train and m ~ num samples in test
    lm = repeat(labels, 1, size(pdm,2)) # label matrix
    spdm = sortperm(pdm, dims=1)

    lm = lm[spdm] # sorted lm according to distances in pdm
    return mean(lm[1:k, :], dims=1) # returns probability of class probability for each test point
end

function knn_probs_all(pdm::Matrix, labels::Union{Vector, SubArray}) # not optimal function in any way!!!
    # pdm ~ n x m, where n ~ num samples in train and m ~ num samples in test
    lm = repeat(labels, size(pdm,2), 1) # label matrix
    spdm = sortperm(pdm, dims=1)

    lm = lm[spdm] # sorted lm according to distances in pdm
    prob_mat = cumsum(lm, dims=1) ./ collect(1:size(pdm, 1))
    return prob_mat
end

function knn_predict_multiclass(pdm::Matrix, labels::Union{Vector, SubArray})
    # pdm ~ n x m, where n ~ num samples in train and m ~ num samples in test
    lm = repeat(labels, size(pdm,2), 1) # label matrix
    spdm = sortperm(pdm, dims=1)

    lm = lm[spdm]
    # 
    predicted = cat([[countunique(lm[1:i,j]) for i=1:size(lm,1)] for j=1:size(lm,2)]..., dims=2)
    return predicted
end

function gram_matrix(x₁, x₂, metric; verbose::Bool=true, wandb_progress::Bool=false)
    lx₂ = length(x₂)
    js = (verbose) ? tqdm(1:lx₂) : 1:lx₂
    is = 1:length(x₁)
    dist_matrix = zeros(length(x₁), length(x₂))
    for j in js
        Threads.@threads for i in is
            dist_matrix[i, j] = Flux.mean(metric(x₂[j], x₁[i]))
        end
        if wandb_progress
            try
                Wandb.log(lg, Dict("GramMatrix/Progress"=> round(j/lx₂, 2)*100))
            catch 
                @warn "error"
            end
        end
    end
    return dist_matrix
end
