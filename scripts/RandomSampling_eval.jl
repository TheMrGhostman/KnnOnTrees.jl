using ArgParse, DrWatson, BSON, DataFrames, Random, Serialization
using Flux, Zygote, Mill, Statistics, KnnOnTrees, LinearAlgebra
using Wandb, Dates, Logging, ProgressBars
using HMillDistance, LIBSVM # SVM

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

#bag_metric, card_metric, iters, batch_size, dataset = "WassersteinMultiset", "MaxCard", 1, 2, "hepatitis"
#batch_size, iters = 2, 1

run_name = "RS-$(dataset)-seed=$(seed)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",#"Julia-testing",
                 name = run_name,
                 config = Dict("transformation" => "Softplus",
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
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=10);


# bag_metric and card_metric switch
bag_m = getfield(HMillDistance, Symbol(bag_metric))
card_m = getfield(HMillDistance, Symbol(card_metric)) 
# metric
_metric = reflectmetric(train[1][1]; set_metric=bag_m, card_metric=card_m, weight_sampler=randn, weight_transform=softplus)

Random.seed!(ui) # rondom initialization of weights with fiexed seed

θ, f = Flux.destructure(_metric)


metric = mean ∘ _metric;


Random.seed!()

# log parameters in Table
θ_names = destructure_metric_to_ws(metric)
θ_best = Flux.destructure(metric)[1]
parameters = Wandb.Table(data=hcat(string.(θ_names[1]), θ_best), columns=["names", "values"])
Wandb.log(lg, Dict("parameters_tab"=>parameters,))

#evaluation SVM and KNN
gm_tr = Symmetric(gram_matrix(train[1], train[1], metric, verbose=false));
gm_val = gram_matrix(train[1], val[1], metric, verbose=false);
gm_tst = gram_matrix(train[1], test[1], metric, verbose=false);

rbfkernel(x, γ) = exp.(- abs.(x) ./ γ)


res = []
for γ ∈ tqdm(0.1:0.1:20) # 0.01
    model = svmtrain(rbfkernel(gm_tr, γ), train[2]; kernel=LIBSVM.Kernel.Precomputed, verbose=false);

    y_train_pr, _ = svmpredict(model, rbfkernel(gm_tr, γ));
    y_valid_pr, _ = svmpredict(model, rbfkernel(gm_val, γ));
    y_test_pr, _ = svmpredict(model, rbfkernel(gm_tst, γ));

    train_a = mean(y_train_pr .== train[2])
    valid_a = mean(y_valid_pr .== val[2])
    test_a = mean(y_test_pr .== test[2])

    push!(res, [γ, train_a, valid_a, test_a])
end

## logging
svm_matrix = permutedims(hcat(res...), (2,1))
svm_columns = ["γ", "train_acc", "valid_acc", "test_acc"]
SVM_results = Wandb.Table(data=svm_matrix, columns=svm_columns)
accs_plot = Wandb.plot_line_series(svm_matrix[:,1], transpose(svm_matrix[:,2:end]), ["train", "valid", "test"], "Accuracy", "γ")
_,argmax_ = findmax(svm_matrix[:,end])

Wandb.log(lg, Dict("SVM_plot"=>accs_plot, "SVM_tab"=>SVM_results))
update_config!(lg, Dict("SVM-(γ|t|v|t)" => round.(svm_matrix[argmax_, :], digits=3), "SVM-max"=>round(svm_matrix[argmax_, end], digits=4)))


# KNN
val_probs = knn_predict_multiclass(gm_val, train[2]);
tst_probs = knn_predict_multiclass(gm_tst, train[2]);

tr_len = length(train[2]);
accuracy_val = mean(val_probs .== repeat(val[2], 1, tr_len)', dims=2)[:];
accuracy_tst = mean(tst_probs .== repeat(test[2], 1, tr_len)', dims=2)[:];

## logging
knn_matrix = hcat(1:tr_len, accuracy_val, accuracy_tst)
knn_columns=["k", "valid_acc", "test_acc"]
accs_plot2 = Wandb.plot_line_series(collect(1:tr_len), hcat(accuracy_val, accuracy_tst)', ["valid", "test"], "Accuracy", "k")
KNN_results = Wandb.Table(data=knn_matrix, columns=knn_columns)
_,argmax_ = findmax(knn_matrix[:,end])

Wandb.log(lg, Dict("KNN_plot"=>accs_plot2, "KNN_tab"=>KNN_results))
update_config!(lg, Dict("KNN-(k|v|t)" => round.(knn_matrix[argmax_, :], digits=3), "KNN-max"=>round(knn_matrix[argmax_, end], digits=4)))

# Finish the run (Logger)
close(lg)


id = (seed=seed, ui=ui, reg=reg)
savedir = datadir("triplet", dataset, "$(seed)") 
results = (
    model=metric, 
    metric=_metric, 
    seed=seed, 
    params=θ_best, 
    param_names=θ_names,
    iters=iters, 
    learning_rate=learning_rate, 
    batch_size=batch_size, 
    history=history, 
    train=train, 
    val=val, 
    test=test, 
    ui=id[:ui], 
    svm_res = DataFrame(svm_matrix, svm_columns),
    knn_res = DataFrame(knn_matrix, knn_columns),
    homogen_depth = homogen_depth,
    bag_metric=bag_metric,
    card_metric=card_metric,
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