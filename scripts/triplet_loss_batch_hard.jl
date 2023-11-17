using ArgParse, DrWatson, BSON, DataFrames, Random
using Flux, Zygote, Mill, Statistics, KnnOnTrees
using Wandb, Dates, Logging, ProgressBars
using HMillDistance

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
    "ui"
        arg_type = Int
        help = "unique identifier"
        default = Int(rand(1:1e8)) # test for error
    "log_pars"
        arg_type = Int
        help = "if we want to track progres of parameter updates -> 0/1 ≈ false/true"
        default = 0
end

parsed_args = parse_args(ARGS, s)
@unpack dataset, seed, iters, learning_rate, batch_size, ui, log_pars = parsed_args
# dataset, seed, iters, learning_rate, batch_size, ui = "Mutagenesis", 666, 1000, 1e-2, 10, 111


run_name = "TripletLoss-BH-$(dataset)-seed=$(seed)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",#"Julia-testing",
                 name = run_name,
                 config = Dict("learning_rate" => learning_rate,
                               "batch_size" => batch_size,
                               "transformation" => "Softplus",
                               "initialization" => "randn",
                               "dataset" => dataset,
                               "iters" => iters,
                               "seed" => seed,
                               "ui" => ui))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)


start = time()
data = load_dataset(dataset; to_mill=true);
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed);

# Loss function
#  xₐ, xₚ, xₙ, α ≈ anchor, positive, negative, margin
#triplet_loss(model, xₐ, xₚ, xₙ, α=0) = sum(Flux.mean.(model.(xₐ, xₚ)) .- Flux.mean.(model.(xₐ, xₙ)) .+ α)
max_triplet_loss(model, xₐ, xₚ, xₙ, α=0) = max(mean( model.(xₐ, xₚ) .- model.(xₐ, xₙ) .+ α ), 0)
triplet_loss(model, xₐ, xₚ, xₙ, α=0) = mean( model.(xₐ, xₚ) .- model.(xₐ, xₙ) .+ α )
triplet_accuracy(model, xₐ, xₚ, xₙ) = mean(model.(xₐ, xₚ) .<= model.(xₐ, xₙ)) # Not exactly accuracy
# I assume that possitive and anchor should be closer to each other


# metric
Random.seed!(ui) # rondom initialization of weights with fiexed seed
_metric = reflectmetric(train[1][1]; weight_sampler=randn, weight_transform=softplus)
metric = mean ∘ _metric;
# trainable parameters & optimizer
ps = Flux.params(metric);
opt = ADAM(learning_rate) #opt_state = Flux.setup(ADAM(), metric);
Random.seed!()

#margin 
α = 1f0

history = Dict("Training/Loss"=>[], "Training/TripletAccuracy"=>[], "Validation/Loss"=>[],"Validation/TripletAccuracy"=>[])

for iter ∈ tqdm(1:iters)
    batch_ = randperm(length(train[2]))[1:batch_size]
    xₐ, xₚ, xₙ = OfflineBatchHardTriplets(metric, train[1][batch_], train[2][batch_])#SampleTriplets(train..., batch_size, true);
    # Gradients ≈ Forward + Backward
    loss_, grad = Flux.withgradient(() -> max_triplet_loss(metric, xₐ, xₚ, xₙ, α), ps);
    # Optimization step
    Flux.update!(opt, ps, grad)
    # Logging training
    acc_ = triplet_accuracy(metric, xₐ, xₚ, xₙ)
    if mod(iter, 20)==0
        xₐᵥ, xₚᵥ, xₙᵥ = SampleTriplets(val..., length(val[2]), false); # There is sampling too
        v_loss = triplet_loss(metric, xₐᵥ, xₚᵥ, xₙᵥ, α); # Just approximation -> correlates with choices of xₐᵥ, xₚᵥ, xₙᵥ 
        v_acc = triplet_accuracy(metric, xₐᵥ, xₚᵥ, xₙᵥ);
        loss_dict = Dict("Training/Loss"=>loss_, "Training/TripletAccuracy"=>acc_,"Validation/Loss"=>v_loss, "Validation/TripletAccuracy"=>v_acc)
        if Bool(log_pars)
            par_vec = softplus.(Flux.destructure(metric.inner)[1])'
            par_vec_dict = Dict("Param/no. $(key)"=>value for (key, value) in enumerate(par_vec))
            Wandb.log(lg, merge(loss_dict, par_vec_dict),);
        else
            Wandb.log(lg, loss_dict,);
        end
        push!(history["Training/Loss"], loss_)
        push!(history["Training/TripletAccuracy"], acc_)
        push!(history["Validation/Loss"], v_loss)
        push!(history["Validation/TripletAccuracy"], v_acc)
    else
        Wandb.log(lg, Dict("Training/Loss"=>loss_, "Training/TripletAccuracy"=>acc_),);
        push!(history["Training/Loss"], loss_)
        push!(history["Training/TripletAccuracy"], acc_)
    end
    (isnan(loss_)) ? break : continue 
end

# Finish the run (Logger)
close(lg)

id = (seed=seed, ui=ui)
savef = joinpath(datadir("triplet", dataset, "$(seed)"), "$(run_name).bson");
results = (
    model=metric, metric=_metric, seed=seed, params=ps, iters=iters, 
    learning_rate=learning_rate, batch_size=batch_size, history=history, 
    train=train, val=val, test=test, ui=id[:ui]
)

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"
