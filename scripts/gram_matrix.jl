using ArgParse, DrWatson, Base.Threads
using CSV, BSON, DataFrames, Distributions 
using Mill, GHMill
using KnnOnTrees

using Flux, Zygote


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

println("threads on run: $(nthreads())")

#dataset, seed, table_idx = "Mutagenesis", 2 , 1392

#file = BSON.load(datadir("knn_rs_ours", "clf", "Mutagenesis", string(seed), "knn_seed=$(seed)_ui=$(table_idx).bson"))

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


pdm_train = knn(train[1], train[1], W, dummy_encoder, structure["*"]; verbose=true);
pdm_valid = knn(train[1], val[1], W, dummy_encoder, structure["*"]; verbose=true);
pdm_test  = knn(train[1], test[1], W, dummy_encoder, structure["*"]; verbose=true);

id = (seed=seed, ui=table_idx)
savef = joinpath(datadir("Gram_matrix", "clf", dataset, "$(seed)"), savename("gram", id, "bson", digits=5));

results = (
    weights = W,
    pdm_train = pdm_train, 
    pdm_valid = pdm_valid,
    pdm_test = pdm_test,
    y_train = train[2],
    y_valid = val[2],
    y_test = test[2],
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
