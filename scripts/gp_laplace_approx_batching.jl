#using Revise
using ArgParse, DrWatson, BSON, DataFrames, Random, Serialization
using Flux, Zygote, Mill, Statistics, KnnOnTrees, LinearAlgebra, EvalMetrics
using Wandb, Dates, Logging, ProgressBars, BenchmarkTools
using HMillDistance, Optim

using AbstractGPs, GPLikelihoods, Distributions, KernelFunctions, ApproximateGPs
using Plots, LaTeXStrings

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
    "learning_rate"        
        arg_type = Float64
        help = "Learning rate for optimizer"
        default = 0.01
    "batch_size"        
        arg_type = Int
        help = "Batch size"
        default = 10
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
@unpack dataset, seed, iters, learning_rate, batch_size, kernel, gamma, ui, homogen_depth, bag_metric, card_metric = parsed_args
#dataset, seed, iters, kernel, gamma, ui = "Mutagenesis", 666, 1, "Laplacian", "nontrainable",1001
trainable_ = gamma == "trainable";
@info parsed_args

run_name = "LatentGP-LAB-$(dataset)-seed=$(seed)-kernel=$(kernel)-gamma=$(gamma)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",
                 name = run_name,
                 config = Dict("learning_rate" => learning_rate,
                               "batch_size" => batch_size,
                               "kernel" => kernel,
                               "transformation" => "Softplus",
                               "initialization" => "0.54 ⋅ ones",
                               "gamma" => gamma,
                               "dataset" => dataset,
                               "homogen_depth" => homogen_depth,
                               "bag_metric" => bag_metric,
                               "card_metric" => card_metric,
                               "iters" => iters,
                               "seed" => seed,
                               "ui" => ui))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)

start = time()
data = load_dataset(dataset; to_mill=true, to_pad_leafs=false, depth=homogen_depth);
data[2] .= binary_class_transform(data[2], (0,1)); # GP implementation require classes (0,1)
#data = (bag_metric == "WassersteinMultiset") ? (HMillDistance.pad_leaves_for_wasserstein.(data[1]), data[2]) : data; #TODO think about this
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=10);

# bag_metric and card_metric switch
bag_m = getfield(HMillDistance, Symbol(bag_metric))
card_m = getfield(HMillDistance, Symbol(card_metric)) 
# 1) define metrics
metric = reflectmetric(data[1][1], set_metric=bag_m, card_metric=card_m, weight_sampler=x->0.54.*ones(x), weight_transform=softplus)
# 2) specify kernel
KernelConstructor_ = KernelSelector(kernel; trainable=trainable_)
Kernel = KernelConstructor_(metric)
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

# 5) create proper batches and proper objectives 
# i try to split data into disjunct parts so i can then fuse them together before prediction and not lose cache
batches_idxes = KnnOnTrees.BalancedDisjunctBinaryBatches(train[2], batch_size);
batches = [(train[1][b_idx], train[2][b_idx]) for b_idx ∈ batches_idxes];
objectives = [build_laplace_objective(build_latent_gp, batch...) for batch in batches];
n_obj = length(objectives)
# 6) optimise
opt = ADAM(learning_rate)

# 7) training loop
history = Dict("Training/Loss"=>[])

pbar = ProgressBar(1:iters);
for iter ∈ 1:round(iters/n_obj)
    obj_loss = 0
    for objective ∈ objectives
        loss, grads = Zygote.withgradient(θ -> objective(θ), θ_init)
        Flux.Optimise.update!(opt, θ_init, grads[1])
        #logging
        Wandb.log(lg, Dict("Training/Loss"=>loss,),);
        push!(history["Training/Loss"], loss)
        update(pbar)
        obj_loss += loss
    end
    Wandb.log(lg, Dict("Training/Loss_sum"=>obj_loss,),);
end

# "rename" parameters
θ_best = deepcopy(θ_init)
#log parse
parameters = Wandb.Table(data=hcat(string.(θ_names[1]),θ_best), columns=["names", "values"])
Wandb.log(lg, Dict("parameters_tab"=>parameters,))
println("I am here!!!!!!")
# 9) build GP
lf = build_latent_gp(θ_best)
# 10) merge batches and caches
X = vcat(getindex.(batches, 1)...)
Y = vcat(getindex.(batches, 2)...)
f_cached = vcat([obj.cache.f for obj ∈ objectives]...)
# 11) compute posterior
f_post = posterior(LaplaceApproximation(; f_init=f_cached), lf(X), Y)
ŷₜᵣ = lf.lik.invlink.(mean(f_post(X)))

# predict validation set (FiniteGP)
fxᵥ = f_post(val[1], 1e-8)
ŷᵥ = lf.lik.invlink.(mean(fxᵥ))
# predict testing set
fxₜ = f_post(test[1], 1e-8)
ŷₜ = lf.lik.invlink.(mean(fxₜ))

# compute metrics
auc_tr= auc_trapezoidal(prcurve(Y, ŷₜᵣ)...) # ad auc
auc_val= auc_trapezoidal(prcurve(val[2], ŷᵥ)...) # ad auc
auc_tst= auc_trapezoidal(prcurve(test[2], ŷₜ)...) # ad auc

acc_tr = mean(Y .== (ŷₜᵣ .>= 0.5))
acc_val = mean(val[2] .== (ŷᵥ .>= 0.5))
acc_tst = mean(test[2] .== (ŷₜ .>= 0.5))

update_config!(lg, Dict(
    "auc_train" => round(auc_tr, digits=3), 
    "auc_val" => round(auc_val, digits=3),
    "auc_test" => round(auc_tst, digits=3),
    "acc_train" => round(acc_tr, digits=5), 
    "acc_val" => round(acc_val, digits=5),
    "acc_test" => round(acc_tst, digits=5),
    )
);



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
    ui=id[:ui],  
    # posterior distributions and fitine GPs
    train_post=f_post, 
    valid_post=fxᵥ, 
    test_post=fxₜ,
    # data splits 
    train=(X,Y),
    orig_train = train,
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

close(lg)