#using Revise
using ArgParse, DrWatson, BSON, DataFrames, Random, Tables
using Flux, Zygote, Mill, Statistics, KnnOnTrees, LinearAlgebra, EvalMetrics
using Wandb, Dates, Logging, ProgressBars, BenchmarkTools
using HMillDistance, Optim

using AbstractGPs, GPLikelihoods, Distributions, KernelFunctions, ApproximateGPs
using Plots, LaTeXStrings

function wandb_log_callback(x, logger_, class)
	Wandb.log(logger_, Dict("Training-$(class)/Loss"=>x.value, "Training-$(class)/GradNorm"=>x.g_norm))
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
        default = Int(rand(1:1e8)) # test for error
end

parsed_args = parse_args(ARGS, s)
@unpack dataset, seed, iters, kernel, gamma, ui = parsed_args
#dataset, seed, iters, kernel, gamma, ui = "chess", 666, 1, "Laplacian", "nontrainable",1001
trainable_ = gamma == "trainable";

run_name = "LatentGP-LA-$(dataset)-seed=$(seed)-kernel=$(kernel)-gamma=$(gamma)--ui=$(ui)"
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
                               "seed" => seed,
                               "ui" => ui))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)


start = time()
data = load_dataset(dataset; to_mill=true);
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=10);

unique_classes = sort(unique(train[2]))
training_history = Dict()

for class ∈ tqdm(unique_classes)
    # 0) transform classes
    y_train = float.(train[2] .== class)
    y_valid = float.(val[2] .== class)
    y_test = float.(test[2] .== class)

    # 1) define metrics
    metric = reflectmetric(data[1][1], weight_sampler=x->0.54.*ones(x), weight_transform=softplus)
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
    objective = build_laplace_objective(build_latent_gp, train[1], y_train)

    # 5) training of GP
    training_results = Optim.optimize(
        objective, θ -> only(Zygote.gradient(objective, θ)), 
        θ_init, 
        LBFGS(), 
        Optim.Options(show_trace=true, iterations = iters, callback = x->wandb_log_callback(x, lg, class), store_trace = false); 
        inplace=false #, callback=print_iter
    ) 

    # get best/optimized parameters
    θ_best = training_results.minimizer
    lf = build_latent_gp(θ_best)
    # compute approximate posterior on training data
    f_post = posterior(LaplaceApproximation(; f_init=objective.cache.f), lf(train[1]), y_train)
    ŷₜᵣ = lf.lik.invlink.(mean(f_post(train[1])))

    # predict validation set (FiniteGP)
    fxᵥ = f_post(val[1], 1e-8)
    ŷᵥ = lf.lik.invlink.(mean(fxᵥ))
    # predict testing set
    fxₜ = f_post(test[1], 1e-8)
    ŷₜ = lf.lik.invlink.(mean(fxₜ))

    # compute metrics
    auc_tr = try auc_trapezoidal(prcurve(y_train, ŷₜᵣ)...); catch ; 0 end # ad auc
    auc_val = try auc_trapezoidal(prcurve(y_valid, ŷᵥ)...); catch ; 0 end
    auc_tst = try auc_trapezoidal(prcurve(y_test, ŷₜ)...); catch ; 0 end

    acc_tr = mean(y_train .== (ŷₜᵣ .>= 0.5));
    acc_val = mean(y_valid .== (ŷᵥ .>= 0.5));
    acc_tst = mean(y_test .== (ŷₜ .>= 0.5));

    # basic logging per class
    training_history["c=$(class)-model"] = lf
    training_history["c=$(class)-kernel"] = m_st(θ_best)
    training_history["c=$(class)-metric"] = m_st(θ_best).d
    training_history["c=$(class)-params"] = θ_best
    training_history["c=$(class)-history"] = training_results
    # posterior distributions and fitine GPs
    training_history["c=$(class)-train_post"] = f_post
    training_history["c=$(class)-valid_post"] = fxᵥ
    training_history["c=$(class)-test_post"] = fxₜ
    # predictions
    training_history["c=$(class)-y_train"] = ŷₜᵣ
    training_history["c=$(class)-y_valid"] = ŷᵥ
    training_history["c=$(class)-y_test"] = ŷₜ
    # metrics
    training_history["c=$(class)-auc"] = (train=auc_tr, valid=auc_val, test=auc_tst)
    training_history["c=$(class)-accuracy"] = (train=acc_tr, valid=acc_val, test=acc_tst)
end

# global logging
y_train_multi = getindex.(argmax(softmax(hcat([training_history["c=$(class)-y_train"] for class in unique_classes]...), dims=2), dims=2), 2);
y_valid_multi = getindex.(argmax(softmax(hcat([training_history["c=$(class)-y_valid"] for class in unique_classes]...), dims=2), dims=2), 2);
y_test_multi  = getindex.(argmax(softmax(hcat([training_history["c=$(class)-y_test" ] for class in unique_classes]...), dims=2), dims=2), 2);

# turn it into original class index
y_train_multi = getindex(unique_classes, y_train_multi);
y_valid_multi = getindex(unique_classes, y_valid_multi);
y_test_multi = getindex(unique_classes, y_test_multi);

acc_tr = mean(train[2] .== y_train_multi);
acc_val = mean(val[2] .== y_valid_multi);
acc_tst = mean(test[2] .== y_test_multi);

auc_tr, auc_val, auc_tst = Tables.columntable([values(training_history["c=$(class)-auc"]) for class in unique_classes]);
auc_tr, auc_val, auc_tst = mean(auc_tr), mean(auc_val), mean(auc_tst)

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
savef = joinpath(datadir("GPs", dataset, "$(seed)"), "$(run_name).bson");
results = (
    # basic log
    kernel_type=kernel,
    seed=seed, 
    iters=iters, 
    ui=id[:ui],  
    # data splits 
    train=train,
    val=val,
    test=test, 
    # metrics
    auc = (train=auc_tr, valid=auc_val, test=auc_tst),
    accuracy = (train=acc_tr, valid=acc_val, test=acc_tst),
)

result = Dict{Symbol, Any}([Symbol(sym)=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
result = merge(training_history, result);
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"


