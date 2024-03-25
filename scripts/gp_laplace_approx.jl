#using Revise
using ArgParse, DrWatson, BSON, DataFrames, Random, Serialization
using Flux, Zygote, Mill, Statistics, KnnOnTrees, LinearAlgebra, EvalMetrics
using Wandb, Dates, Logging, ProgressBars, BenchmarkTools
using HMillDistance, Optim

using AbstractGPs, GPLikelihoods, Distributions, KernelFunctions, ApproximateGPs
using Plots, LaTeXStrings

function wandb_log_callback(x, logger_)
	Wandb.log(logger_, Dict("Training/Loss"=>x.value, "Training/GradNorm"=>x.g_norm))
	return false
end

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
    "iters"
        arg_type = Int
        help = "Number or iterations"
        default = 1000
    "kernel"        
        arg_type = String
        help = "Select Kernel option: (\"Laplacian\", \"Gaussian\", \"Matern32\") "
        default = "Laplacian"
    "gamma"
        arg_type = String
        help = "let γ parameter trainable \"trainable\" or \"nontrainable\" "
        default = "nontrainable"
    "ui"
        arg_type = Int
        help = "unique identifier"
        default = Int(rand(1:1e8))
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
@unpack dataset, seed, iters, kernel, gamma, ui, homogen_depth, bag_metric, card_metric = parsed_args
#dataset, seed, iters, kernel, gamma, ui = "Mutagenesis", 666, 1, "Laplacian", "nontrainable",1001
trainable_ = gamma == "trainable";

run_name = "LatentGP-LA-$(dataset)-seed=$(seed)-kernel=$(kernel)-gamma=$(gamma)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",
                 name = run_name,
                 config = Dict(
                               "kernel" => kernel,
                               "transformation" => "Softplus",
                               "initialization" => "0.54 ⋅ ones",
                               "gamma" => gamma,
                               "dataset" => dataset,
                               "iters" => iters,
                               "homogen_depth" => homogen_depth,
                               "bag_metric" => bag_metric,
                               "card_metric" => card_metric,
                               "seed" => seed,
                               "ui" => ui))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)


start = time()
data = load_dataset(dataset; to_mill=true, depth=homogen_depth);
data[2] .= binary_class_transform(data[2], (0,1)); # GP implementation require classes (0,1)
#data = (bag_metric == "WassersteinMultiset") ? (HMillDistance.pad_leaves_for_wasserstein.(data[1]), data[2]) : data;
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=0);


# bag_metric and card_metric switch
bag_m = getfield(HMillDistance, Symbol(bag_metric))
card_m = getfield(HMillDistance, Symbol(card_metric)) 
# 1) define metrics
metric = reflectmetric(data[1][1], set_metric=bag_m, card_metric=card_m, weight_sampler=x->0.54.*ones(x), weight_transform=softplus)
# 2) specify kernel
KernelConstructor_ = KernelSelector(kernel; trainable=trainable_) #LaplacianHMillKernel # TODO add more options | Laplacian is the first
Kernel = KernelConstructor_(metric)#; γ=1.0, trainable=trainable_)
# 3) initialize θ
θ_init, m_st = Flux.destructure(Kernel)
θ_names = destructure_metric_to_ws(Kernel.d);

# 4) build_latent_gp function 
function build_latent_gp(θ)    
    θ_, f_ = Flux.destructure(Kernel)
    kernel_ = f_(θ) #KernelConstructor(f_(θ)) 
    dist_y_given_f = BernoulliLikelihood()  # has logistic invlink by default
    jitter = 1e-3  # required for numeric stability
    return LatentGP(GP(kernel_), dist_y_given_f, jitter)
end;

# 4) build_laplace_objective
objective = build_laplace_objective(build_latent_gp, train...)

# 5) training of GP
training_results = Optim.optimize(
    objective, θ -> only(Zygote.gradient(objective, θ)), 
    θ_init, 
    LBFGS(), 
    Optim.Options(show_trace=true, iterations = iters, callback = x->wandb_log_callback(x, lg), store_trace = false); 
    inplace=false #, callback=print_iter
)

# get best/optimized parameters
θ_best = training_results.minimizer
# quickly log parameters
parameters = Wandb.Table(data=hcat(string.(θ_names[1]),θ_best), columns=["names", "values"])
Wandb.log(lg, Dict("parameters_tab"=>parameters,))

lf = build_latent_gp(θ_best)
# compute approximate posterior on training data
f_post = posterior(LaplaceApproximation(; f_init=objective.cache.f), lf(train[1]), train[2])
ŷₜᵣ = lf.lik.invlink.(mean(f_post(train[1])))

# predict validation set (FiniteGP)
fxᵥ = f_post(val[1], 1e-8)
ŷᵥ = lf.lik.invlink.(mean(fxᵥ))
# predict testing set
fxₜ = f_post(test[1], 1e-8)
ŷₜ = lf.lik.invlink.(mean(fxₜ))

# compute metrics
auc_tr= auc_trapezoidal(prcurve(train[2], ŷₜᵣ)...); # ad auc
auc_val= auc_trapezoidal(prcurve(val[2], ŷᵥ)...); # ad auc
auc_tst= auc_trapezoidal(prcurve(test[2], ŷₜ)...); # ad auc

acc_tr = mean(train[2] .== (ŷₜᵣ .>= 0.5));
acc_val = mean(val[2] .== (ŷᵥ .>= 0.5));
acc_tst = mean(test[2] .== (ŷₜ .>= 0.5));

update_config!(lg, Dict(
    "auc_train" => round(auc_tr, digits=3), 
    "auc_val" => round(auc_val, digits=3),
    "auc_test" => round(auc_tst, digits=3),
    "acc_train" => round(acc_tr, digits=5), 
    "acc_val" => round(acc_val, digits=5),
    "acc_test" => round(acc_tst, digits=5),
    )
);

close(lg)

id = (seed=seed, ui=ui, kernel=kernel)
savedir = datadir("GPs", dataset, "$(seed)")
results = (
    # basic log
    model=lf, # LatentGP
    kernel=m_st(θ_best), 
    metric=m_st(θ_best).d,
    seed=seed, 
    params=θ_best, 
    iters=iters, 
    history=training_results,
    ui=id[:ui],  
    # posterior distributions and fitine GPs
    train_post=f_post, 
    valid_post=fxᵥ, 
    test_post=fxₜ,
    # data splits 
    train=train,
    val=val,
    test=test, 
    # predictions
    y_train = ŷₜᵣ,
    y_valid = ŷᵥ,
    y_test = ŷₜ,
    # metrics
    auc = (train=auc_tr, valid=auc_val, test=auc_tst),
    accuracy = (train=acc_tr, valid=acc_val, test=acc_tst),
)

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
if !ispath(savedir)
    mkpath(savedir)
end
serialize(joinpath(savedir, "$(run_name).jls"), result) 
#tagsave(joinpath(savedir, "$(run_name).bson"), result, safe = true);
@info "Results were saved into file $(savedir) --- $(run_name) (.bson / .jls)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"
println("Results were saved into file $(savedir) --- $(run_name) (.bson / .jls)")


