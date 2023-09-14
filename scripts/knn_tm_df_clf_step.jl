using ArgParse, DrWatson
using BSON, Statistics
using Mill
using KnnOnTrees
using TreeMetrics.NaiveDistance

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        arg_type = String
        help = "Name of dataset to use"
    "seed"
        arg_type = Int
        help = "Random seed for initialization of data splits"
        default = 666
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, seed = parsed_args
# dataset, seed = "Mutagenesis", 666

start = time()
data = load_dataset(dataset; to_mill=false);
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed);
#(X_tr, y_tr), (X_val, y_val), (X_tst, y_tst)


# run knn and return pairwise distance matrix
start_pdm=time()
pdm_val = knn_tm(train[1], val[1], (x,y)->NaiveDistance.dist(x,y, smoothdist); verbose=true);
pdm_tst = knn_tm(train[1], test[1], (x,y)->NaiveDistance.dist(x,y, smoothdist); verbose=true);
end_pdm = time()
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

id = (seed=seed, ui=1)
savef = joinpath(datadir("knn_rs_tm", "clf", dataset, "$(seed)"), savename("knn_tm", id, "bson", digits=5));

results = (
    pdm_val = pdm_val,
    pdm_tst = pdm_tst,
    y_val = val[2],
    y_tst = test[2],
    accuracy_val = accuracy_val,
    accuracy_tst = accuracy_tst,
    seed=seed,
    dataset=dataset,
    train_time = end_pdm - start_pdm,
    ui=id[:ui],
);

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"