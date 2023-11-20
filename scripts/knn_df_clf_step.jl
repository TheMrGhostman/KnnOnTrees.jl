using ArgParse, DrWatson, Base.Threads
using CSV, BSON, DataFrames, Distributions 
using Mill, GHMill
using KnnOnTrees

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        arg_type = String
        help = "Name of dataset to use"
    "table_idx"
        arg_type = Int
        help = "index in table of all possible samples"
    "seed"
        arg_type = Int
        help = "Random seed for initialization of data splits"
        default = 666
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, table_idx, seed = parsed_args
# dataset, seed = "Mutagenesis", 666

println("threads on run: $(nthreads())")

start = time()
data = load_dataset(dataset; to_mill=true);
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed);
#(X_tr, y_tr), (X_val, y_val), (X_tst, y_tst)

# initialize dummy_encoder and Weights dict
dummy_encoder = GHMill.reflectinencoder(train[1][1]; verbose=false);
structure, W_init = GHMill.tree_struct_and_weights_dict(dummy_encoder; rwi=true);

# load/sample weights
tab_path = datadir("rs_options", "$(dataset).csv")
df = CSV.read(tab_path, DataFrame);
W = sample_weights(W_init, df, table_idx)
#W = sample_weights(W_init, MixtureModel([Uniform(0,100), Exponential(log(2)*30)], [0.35, 0.65]))
# W = sample_weights(W_init, dataframe, index)

## Other adequate options
# Uniform(0,100) 
# Exponential(log(2)*30)
# MixtureModel([Uniform(0,100), Exponential(log(2)*30)], [0.35, 0.65]) 

# run knn and return pairwise distance matrix
pdm_val = knn(train[1], val[1], W, dummy_encoder, structure["*"]; verbose=true);
pdm_tst = knn(train[1], test[1], W, dummy_encoder, structure["*"]; verbose=true);

# classification part
## we get matrix with with same dimensions as pdm
## but now row index is equivalent to k (number of neighbors)
y_val_pred = knn_predict_multiclass(pdm_val, train[2])
y_tst_pred = knn_predict_multiclass(pdm_tst, train[2])

# Evaluation of results
## Accuracy
tr_len = length(train[2])
accuracy_val = mean(y_val_pred .== repeat(val[2], 1, tr_len)', dims=2)[:]
accuracy_tst = mean(y_tst_pred .== repeat(test[2], 1, tr_len)', dims=2)[:]

id = (seed=seed, ui=table_idx)
savef = joinpath(datadir("knn_rs_ours", "clf", dataset, "$(seed)"), savename("knn", id, "bson", digits=5));

results = (
    weights = W, 
    pdm_val = pdm_val,
    pdm_tst = pdm_tst,
    y_val = val[2],
    y_tst = test[2],
    accuracy_val = accuracy_val,
    accuracy_tst = accuracy_tst,
    seed=seed,
    dataset=dataset,
    ui=id[:ui],
    table_idx=id[:ui]
);

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"