function sample_weights(W_init::Dict, distribution::Distribution)
    keys_ = collect(keys(W_init))
    w_vector = rand(distribution, length(keys_))
    W = Dict{String, Float32}([key => w_vector[idx] for (idx, key) in enumerate(keys_)])
    return W
end

function sample_weights(W_init::Dict, df::DataFrame, index::Int)
    keys = collect(keys(W_init))
    W_row = df[index, keys];
    W = Dict{String, Float32}([key=>W_row[key] for key in keys])
    return W
end

function get_most_occured_class(d::Dict)
    collect(keys(d))[findmax(collect(values(d)))[2]]
end

function get_most_occured_class(d::Dict, uniques::Array)
    #@assert sort(collect(keys(d))) == sort(unqiues) "Something went wrong in get_most_occured_class"
    values_ = map(key->d[key], uniques)
    return uniques[findmax(values_)[2]]
end

function countunique(x)
    cm = StatsBase.countmap(x)
    un = unique(x)
    return get_most_occured_class(cm, un)
end