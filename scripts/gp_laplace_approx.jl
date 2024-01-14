#using Revise
using ArgParse, DrWatson, BSON, DataFrames, Random
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
        help = "Kernel"
        default = "LaplacianHMILKernel"
    "ui"
        arg_type = Int
        help = "unique identifier"
        default = Int(rand(1:1e8)) # test for error
end

parsed_args = parse_args(ARGS, s)
@unpack dataset, seed, iters, kernel, ui = parsed_args
#dataset, seed, iters, kernel, ui = "Mutagenesis", 666, 1, "Laplacian", 1001

run_name = "LatentGP-LA-$(dataset)-seed=$(seed)-kernel=$(kernel)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",
                 name = run_name,
                 config = Dict(
                               "kernel" => kernel,
                               "transformation" => "Softplus",
                               "initialization" => "0.54 ⋅ ones",
                               "dataset" => dataset,
                               "iters" => iters,
                               "seed" => seed,
                               "ui" => ui))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)


start = time()
data = load_dataset(dataset; to_mill=true);
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=0);

# TODO make it properly
train[2][:] = train[2][:] .- 1.0;
val[2][:] = val[2][:] .- 1.0;
test[2][:] = test[2][:] .- 1.0;

# 1) define metrics
metric = reflectmetric(data[1][1], weight_sampler=x->0.54.*ones(x), weight_transform=softplus)

# 2) build_latent_gp function 
function build_latent_gp(θ)    
    θ_, f_ = Flux.destructure(metric)
    kernel_ = LaplacianHMILKernel(f_(θ)) # TODO change kernel option
    dist_y_given_f = BernoulliLikelihood()  # has logistic invlink by default
    jitter = 1e-3  # required for numeric stability
    return LatentGP(GP(kernel_), dist_y_given_f, jitter)
end;

# 3) initialize θ
θ_init, m_st = Flux.destructure(metric)

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

id = (seed=seed, ui=kernel, kernel=kernel)
savef = joinpath(datadir("GPs", dataset, "$(seed)"), "$(run_name).bson");
results = (
    # basic log
    model=lf, # LatentGP
    metric=m_st(θ_best), 
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
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"


