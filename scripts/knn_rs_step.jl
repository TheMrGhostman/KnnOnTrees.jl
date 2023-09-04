using Statistics, MLDatasets, MLUtils, Mill, JsonGrinder, JSON3
using EvalMetrics, Random
using knn_treeloss, GHMill#, Hyperopt
using ArgParse, CSV, BSON, DrWatson, DataFrames

s = ArgParseSettings()
@add_arg_table! s begin
    "table_idx"
        arg_type = Int
        help = "index in table of all possible samples"
    "path"
        arg_type=String
        help = "Path to file with table of samples"
        default = "uniform_rs_table.csv"
    "seed"
        arg_type = Int
        help = "Random seed for initialization of data splits"
        default = 666
end
parsed_args = parse_args(ARGS, s)
@unpack table_idx, path, seed = parsed_args


# load data and extract data
data = MLDatasets.Mutagenesis(split=:all);
sch = JsonGrinder.schema(data.features);
extr = JsonGrinder.suggestextractor(sch);
Xs = extr.(data.features);
Ys = data.targets;

# random split for data --- only train/test --- classification
Random.seed!(seed)
(X_tr, Y_tr), (X_tst, Y_tst) = MLUtils.splitobs((Xs, Ys); at=0.70, shuffle=true);

# prepare dummy model (can be loaded)
dummy_enc = reflectinencoder(Xs[1]; verbose=false);

# initialize structure and W .. weight vector placeholder
W_init = Dict();
structure = identify_w(dummy_enc, Dict(), W_init; verbose=false)["*"];

# load weight vector and save it to W
df = CSV.read(path, DataFrame);
# Any["Kg", "2", "6", "KE", "L*", "Kd", "A", "Ke", "E", "Kf", "I", "KU", "*", "K", "Kk", "Kc"]
W_row = df[table_idx, collect(keys(W_init))];
W = Dict(map(key -> key=>W_row[key], collect(keys(W_init)))); #TODO make it nicer

# run knn and return pairwise distance matrix
pdm = knn(X_tr, X_tst, W, dummy_enc, structure; verbose=true);

# classification part
## we get matrix with with same dimensions as pdm
## but now row index is equivalent to k (number of neighbors)
## and every number on row is probability of sample belonging to class 1
## for given (row index) k
probs_all = knn_probs_all(pdm, Y_tr);
accuracy_ = mean((probs_all .> 0.5) .== repeat(Y_tst, 1, 132)', dims=2)[:]; # accuracy for each k
auc_clf = mapslices(col->auc_trapezoidal(prcurve(getobs(Y_tst), col)...), probs_all, dims=2)[:]; # classification auc

# Anomaly detection 
# filter out anomalies from pdm -> 1 = normal | 0 = anomalous
pdm_ad = pdm[Y_tr .== 1, :];
spdm_ad = sort(pdm_ad, dims=1);
auc_ad = mapslices(col->auc_trapezoidal(prcurve(1 .- getobs(Y_tst), col)...), spdm_ad, dims=2)[:]; # ad auc


# formulate saving name for file
id = (seed=seed, rs = String(split(path, "_")[1]), idx = table_idx)
println(id)
savef = joinpath(datadir("knn_random_search"), savename("knn", id, "bson", digits=5, allowedtypes=(String, Number,)));

# informations to save
clf_v, clf_ci = findmax(auc_clf);
ad_v, ad_ci = findmax(auc_ad);

results = (
    weights = W,
    pdm = pdm,
    class_probs = probs_all,
    class_gt = Array(Y_tst),
    accuracy = accuracy_,
    auc_clf = auc_clf,
    pdm_ad = pdm_ad,
    auc_ad = auc_ad,
    auc_clf_max = clf_v,
    auc_clf_k = clf_ci[1],
    auc_ad_max = ad_v,
    auc_ad_k = ad_ci[1],
    seed=seed,
    rs = id[:rs],
);

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"

# update csv file ... add AUC, k, etc.
