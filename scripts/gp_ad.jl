#using Revise
using ArgParse, DrWatson, BSON, DataFrames, Random, Serialization
using Flux, Zygote, Mill, Statistics, KnnOnTrees, LinearAlgebra
using Wandb, Dates, Logging, ProgressBars, BenchmarkTools, StatsBase
using HMillDistance, Optim, Base.Threads, EvalMetrics

using AbstractGPs, Distributions, KernelFunctions


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
        default = 10
    "learning_rate"        
        arg_type = Float64
        help = "Learning rate for optimizer"
        default = 0.01
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
@unpack dataset, seed, iters, learning_rate, kernel, gamma, ui, homogen_depth, bag_metric, card_metric = parsed_args
#dataset, seed, iters, kernel, gamma, ui = "Mutagenesis", 666, 1, "Laplacian", "nontrainable",1001
trainable_ = gamma == "trainable";
@info "Threads -> $(nthreads())"

run_name = "GP-AD-$(dataset)-seed=$(seed)-kernel=$(kernel)-gamma=$(gamma)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="KnnOnTrees - Anomaly Detection",
                 name = run_name,
                 config = Dict(
                               "kernel" => kernel,
                               "transformation" => "Softplus",
                               "initialization" => "0.54 ⋅ ones",
                               "gamma" => gamma,
                               "dataset" => dataset,
                               "iters" => iters,
                               "learning_rate" =>learning_rate,
                               "homogen_depth" => homogen_depth,
                               "bag_metric" => bag_metric,
                               "card_metric" => card_metric,
                               "seed" => seed,
                               "ui" => ui))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)


start = time()
data = load_dataset(dataset; to_mill=true, to_pad_leafs=false, depth=homogen_depth);

# separate normal class 
class_freq = countmap(data[2])
most_frequent_class = sort(collect(class_freq), rev=true,by=x->getindex(x, 2))[1][1]

normal_data = data[1][data[2] .== most_frequent_class]; # class is imaginary 0
anomalous_data = data[1][data[2] .!= most_frequent_class]; # every other class is anomalous (leave-one-in procedure)

train_n, val_n, test_n = preprocess(normal_data, zeros(length(normal_data)); ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=10);
_, val_a, test_a = preprocess(anomalous_data, ones(length(anomalous_data)); ratios=(0.0,0.5,0.5), procedure=:clf, seed=seed, filter_under=10);

train = Array.(train_n);
val = map((a,b)->vcat(a,b), val_n, val_a);
test = map((a,b)->vcat(a,b), test_n, test_a);


# bag_metric and card_metric switch
bag_m = getfield(HMillDistance, Symbol(bag_metric))
card_m = getfield(HMillDistance, Symbol(card_metric)) 
# 1) define metrics
metric = reflectmetric(data[1][1], set_metric=bag_m, card_metric=card_m, weight_sampler=x->0.54.*ones(x), weight_transform=softplus)
# 2) specify kernel
KernelConstructor_ = KernelSelector(kernel; trainable=trainable_) 
Kernel = KernelConstructor_(metric)#; γ=1.0, trainable=trainable_)
# 3) initialize θ
θ_init, m_st = Flux.destructure(Kernel)
θ_names = destructure_metric_to_ws(Kernel.d);

# 4) build_gp function 
function build_gp(θ)    
    θ_, f_ = Flux.destructure(Kernel)
    kernel_ = f_(θ) #KernelConstructor(f_(θ)) 
    return GP(kernel_)
end;

loss(θ, X) = -logpdf(build_gp(θ)(X, 1e-5), ones(size(X, 1)))

# 6) optimise
opt = ADAM(learning_rate)
# 7) training loop
history = Dict("Training/Loss"=>[], "params_stats"=>[])

sqnorm(x,b=0) = sum(y->abs2(y .- b), x)
param_statistics(θ) = Dict(
    "minimum" => minimum(θ), 
    "q_25" => quantile(θ, 0.25), 
    "median" => median(θ),
    "q_75" => quantile(θ, 0.75),
    "maximum" => maximum(θ)
    )

for iter ∈ tqdm(1:iters)
    loss_, grads = Zygote.withgradient(θ -> loss(θ, train[1]), θ_init)
    Flux.Optimise.update!(opt, θ_init, grads[1])
    stats = param_statistics(softplus.(θ_init))
    Wandb.log(lg, merge(Dict("Training/Loss"=>loss_,), stats));
    push!(history["Training/Loss"], loss_)
    push!(history["params_stats"], stats)
end

# get best/optimized parameters
θ_best = deepcopy(θ_init)
# quickly log parameters
parameters = Wandb.Table(data=hcat(string.(θ_names[1]),θ_best), columns=["names", "values"])
Wandb.log(lg, Dict("parameters_tab"=>parameters,))

# build gp
fx = build_gp(θ_best)(train[1])
# compute posterior
f_post = posterior(fx, ones(size(train[2])))
# compute marginals
μ_v, σ²_v = mean_and_var(f_post(val[1]))
μ_t, σ²_t = mean_and_var(f_post(test[1]))


#marg_val = marginals(f_post(val[1]))
#marg_test = marginals(f_post(test[1]))
# marginals for val data
#μ_v = getproperty.(marg_val, :μ)
#σ_v = getproperty.(marg_val, :σ)
# marginals for test data
#μ_t = getproperty.(marg_test, :μ)
#σ_t = getproperty.(marg_test, :σ)

𝓔_v = - μ_v
𝓔_t = - μ_t

𝓥_v = - σ²_v#σ_v.^2 # original formula is -σ^2 but for 1 normal, 0 anomal -> we have 0 normal, 1 anomal
𝓥_t = - σ²_t#σ_t.^2


# compute AUCs
# score with μ
auc_𝓔_val = auc_trapezoidal(prcurve(val[2], 𝓔_v)...)
auc_𝓔_test = auc_trapezoidal(prcurve(test[2], 𝓔_t)...)
# score with neg var -σ^2
auc_𝓥_val = auc_trapezoidal(prcurve(val[2], 𝓥_v)...)
auc_𝓥_test = auc_trapezoidal(prcurve(test[2], 𝓥_t)...)
# score with σ
auc_σ_val = auc_trapezoidal(prcurve(val[2], σ²_v)...)
auc_σ_test = auc_trapezoidal(prcurve(test[2], σ²_t)...)


Wandb.log(lg, Dict(
    "auc_𝓔_val" => round(auc_𝓔_val, digits=3), 
    "auc_𝓔_test" => round(auc_𝓔_test, digits=3),
    "auc_𝓥_val" => round(auc_𝓥_val, digits=3), 
    "auc_𝓥_test" => round(auc_𝓥_test, digits=3),
    "auc_σ_val" => round(auc_σ_val, digits=3), 
    "auc_σ_test" => round(auc_σ_test, digits=3),
    )
);

id = (seed=seed, ui=ui, kernel=kernel)
savedir = datadir("GPs-AD", dataset, "$(seed)")
results = (
    # basic log
    model=build_gp(θ_best), # 
    kernel=m_st(θ_best), 
    metric=m_st(θ_best).d,
    seed=seed, 
    params=θ_best, 
    iters=iters, 
    history=history,
    ui=id[:ui],  
    # posterior distributions and fitine GPs
    train_post=f_post, 
    # data splits 
    train=train,
    val=val,
    test=test, 
    # predictions
    y_𝓔_valid = 𝓔_v,
    y_𝓔_test = 𝓔_t,
    y_𝓥_valid = 𝓥_v,
    y_𝓥_test = 𝓥_t,
    y_σ_valid = σ²_v,
    y_σ_test = σ²_t,
    # metrics
    auc_𝓔 = (valid=auc_𝓔_val, test=auc_𝓔_test),
    auc_𝓥 = (valid=auc_𝓥_val, test=auc_𝓥_test),
    auc_σ = (valid=auc_σ_val, test=auc_σ_test),
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
