#using Revise
using ArgParse, DrWatson, BSON, DataFrames, Random, Serialization
using Flux, Zygote, Mill, Statistics, LinearAlgebra, Distributions, Base.Threads
using Wandb, Dates, Logging, ProgressBars, StatsBase
using KnnOnTrees, HMillDistance, LIBSVM, EvalMetrics

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        arg_type = String
        help = "Name of dataset to use"
        default = "Mutagenesis"
    "seed"
        arg_type = Int
        help = "Random seed for initialization of data splits"
        default = 666
    "ui"
        arg_type = Int
        help = "unique identifier"
        default = Int(rand(1:1e8)) # test for error
    "homogen_depth"
        arg_type = Int
        help = "Depth of tree which is created from homogenous Graphs (TUDataset). For other data argument is ignored!"
        default = 4
    "bag_metric"
        arg_type = String
        help = "Which type of metric to use for comparision of bags \\
                 options: (\"ChamferDistance\", \"WassersteinProbDist\", \"WassersteinMultiset\", \"Hausdorff\" )"
        default="ChamferDistance"
    "card_metric"
        arg_type = String
        help = "Which type of metric/transfromation to use for cardinality \\
                options: (\"ScaleOne\", \"MaxCard\")"
        default = "ScaleOne" 
end

parsed_args = parse_args(ARGS, s)
@unpack dataset, seed, ui, homogen_depth, bag_metric, card_metric = parsed_args
@info parsed_args
@info "Threads -> $(nthreads())"

#bag_metric, card_metric, iters, batch_size, dataset = "WassersteinMultiset", "MaxCard", 1, 2, "hepatitis"
#batch_size, iters = 2, 1

run_name = "RS-AD-$(dataset)-seed=$(seed)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="KnnOnTrees - Anomaly Detection",
                 name = run_name,
                 config = Dict("transformation" => "identity",
                               "dataset" => dataset,
                               "homogen_depth" => homogen_depth,
                               "bag_metric" => bag_metric,
                               "card_metric" => card_metric,
                               "seed" => seed,
                               "ui" => ui))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)

start = time()
data = load_dataset(dataset; to_mill=true, to_pad_leafs=false, depth=homogen_depth);
data = (bag_metric == "WassersteinMultiset") ? (HMillDistance.pad_leaves_for_wasserstein.(data[1]), data[2]) : data;

# separate normal class 
class_freq = countmap(data[2])
most_frequent_class = sort(collect(class_freq), rev=true,by=x->getindex(x, 2))[1][1]

normal_data = data[1][data[2] .== most_frequent_class]; # class is imaginary 0
anomalous_data = data[1][data[2] .!= most_frequent_class]; # every other class is anomalous (leave-one-in procedure)

train_n, val_n, test_n = preprocess(normal_data, zeros(length(normal_data)); ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=10);
_, val_a, test_a = preprocess(anomalous_data, ones(length(anomalous_data)); ratios=(0.0,0.5,0.5), procedure=:clf, seed=seed, filter_under=10);

train = train_n;
val = map((a,b)->vcat(a,b), val_n, val_a);
test = map((a,b)->vcat(a,b), test_n, test_a);


# bag_metric and card_metric switch
bag_m = getfield(HMillDistance, Symbol(bag_metric))
card_m = getfield(HMillDistance, Symbol(card_metric)) 
# metric
_metric = reflectmetric(train[1][1]; set_metric=bag_m, card_metric=card_m, weight_sampler=randn, weight_transform=identity)
θ, f = Flux.destructure(_metric)
distro = MixtureModel([Uniform(0,100), Exponential(log(2)*10)], [0.10, 0.90]) # very heavy tail up to 100

Random.seed!(ui) # random initialization of weights with fiexed seed
θ_new = rand(distro, size(θ));
_metric = f(θ_new)

metric = mean ∘ _metric;
Random.seed!()

# log parameters in Table
θ_names = destructure_metric_to_ws(metric)
θ_best = Flux.destructure(metric)[1]
parameters = Wandb.Table(data=hcat(string.(θ_names[1]), θ_best), columns=["names", "values"])
Wandb.log(lg, Dict("parameters_tab"=>parameters,))

#Computing Gram Matrix
#gm_tr = Symmetric(gram_matrix(train[1], train[1], metric, verbose=true));
gm_val = gram_matrix(train[1], val[1], metric, verbose=true);
gm_tst = gram_matrix(train[1], test[1], metric, verbose=true);

sgm_val = sort(gm_val, dims=1);
sgm_tst = sort(gm_tst, dims=1);

# compute AUC for evary k
auc_val = mapslices(col->auc_trapezoidal(prcurve(val[2], col)...), sgm_val, dims=2)[:];
auc_test = mapslices(col->auc_trapezoidal(prcurve(test[2], col)...), sgm_tst, dims=2)[:];

## logging
tr_len = length(train[2])
knn_matrix = hcat(1:tr_len, auc_val, auc_test)
knn_columns=["k", "valid_auc", "test_auc"]
accs_plot2 = Wandb.plot_line_series(collect(1:tr_len), hcat(auc_val, auc_test)', ["valid", "test"], "AUC", "k")
KNN_results = Wandb.Table(data=knn_matrix, columns=knn_columns)
Wandb.log(lg, Dict("KNN_plot"=>accs_plot2, "KNN_tab"=>KNN_results))


_, idx_val = findmax(auc_val)
_, idx_tst = findmax(auc_test)

update_config!(lg, Dict(
    "KNN_v-(k|v|t)" => round.(knn_matrix[idx_val, :], digits=3), 
    "KNN_t-(k|v|t)" => round.(knn_matrix[idx_tst, :], digits=3) ,
    "KNN-max"=>round(knn_matrix[idx_tst, end], digits=4)))


id = (seed=seed, ui=ui)
savedir = datadir("RandomSampling-AD", dataset, "$(seed)") 
results = (
    model=metric, 
    metric=_metric, 
    seed=seed, 
    params=θ_new, 
    param_names=θ_names,
    train=train, 
    val=val, 
    test=test, 
    ui=id[:ui], 
    knn_res = DataFrame(knn_matrix, knn_columns),
    homogen_depth = homogen_depth,
    bag_metric=bag_metric,
    card_metric=card_metric,
    gm_val = gm_val,
    gm_test = gm_tst,
)

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
if !ispath(savedir)
    mkpath(savedir)
end
serialize(joinpath(savedir, "$(run_name).jls"), result) 
tagsave(joinpath(savedir, "$(run_name).bson"), result, safe = true);
@info "Results were saved into file $(savedir)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"
println("Results were saved into file $(savedir) --- $(run_name) (.bson / .jls)")


# Finish the run (Logger)
close(lg)