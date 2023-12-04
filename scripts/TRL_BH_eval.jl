using ArgParse, DrWatson, BSON, DataFrames, Random
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
    "reg"
        arg_type = Float64
        help = "scale for parameter regularization (added to triplet loss)"
        default = 0.0
    "gamma"
        arg_type = Float64
        help = "meen value that regularization is pushed to"
        default = 0.0
    "margin"
        arg_type = Float64
        help = "margin value for triplet loss"
        default = 1.0
    "ui"
        arg_type = Int
        help = "unique identifier"
        default = Int(rand(1:1e8)) # test for error
    "log_pars"
        arg_type = Int
        help = "if we want to track progres of parameter updates -> 0/1 ≈ false/true"
        default = 1 
end

parsed_args = parse_args(ARGS, s)
@unpack dataset, seed, iters, learning_rate, batch_size, reg, gamma, margin, ui, log_pars = parsed_args
# dataset, seed, iters, learning_rate, batch_size, ui = "Mutagenesis", 666, 1000, 1e-2, 10, 111


run_name = "TripletLoss-BH-$(dataset)-seed=$(seed)-ui=$(ui)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",#"Julia-testing",
                 name = run_name,
                 config = Dict("learning_rate" => learning_rate,
                               "batch_size" => batch_size,
                               "transformation" => "Softplus",
                               "initialization" => "randn",
                               "reg" => reg,
                               "gamma" => gamma,
                               "margin"=> margin,
                               "dataset" => dataset,
                               "iters" => iters,
                               "seed" => seed,
                               "ui" => ui,
                               "log_pars"=>log_pars))

# Use LoggingExtras.jl to log to multiple loggers together
global_logger(lg)


start = time()
data = load_dataset(dataset; to_mill=true);
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed, filter_under=0);

# Loss function
#  xₐ, xₚ, xₙ, α ≈ anchor, positive, negative, margin
#triplet_loss(model, xₐ, xₚ, xₙ, α=0) = sum(Flux.mean.(model.(xₐ, xₚ)) .- Flux.mean.(model.(xₐ, xₙ)) .+ α)
sqnorm(x,b=0) = sum(y->abs2(y .- b), x)
max_triplet_loss(model, xₐ, xₚ, xₙ, α=0) = max(mean( model.(xₐ, xₚ) .- model.(xₐ, xₙ) .+ α ), 0) 
reg_max_triplet_loss(model, xₐ, xₚ, xₙ, α=0, β=0, γ=0) = max(mean( model.(xₐ, xₚ) .- model.(xₐ, xₙ) .+ α ), 0) + β .* sqrt(sum(x->sqnorm(x,γ), Flux.params(metric)))
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
α = get_config(lg, "margin")
β = get_config(lg, "reg")
γ = get_config(lg, "gamma")

history = Dict("Training/Loss"=>[], "Training/TripletAccuracy"=>[], "Validation/Loss"=>[],"Validation/TripletAccuracy"=>[])

for iter ∈ tqdm(1:iters)
    batch_ = randperm(length(train[2]))[1:batch_size]
    xₐ, xₚ, xₙ = OfflineBatchHardTriplets(metric, train[1][batch_], train[2][batch_])#SampleTriplets(train..., batch_size, true);
    # Gradients ≈ Forward + Backward
    loss_, grad = Flux.withgradient(() -> reg_max_triplet_loss(metric, xₐ, xₚ, xₙ, α, β, γ), ps);
    # Optimization step
    Flux.update!(opt, ps, grad)
    # Logging training
    acc_ = triplet_accuracy(metric, xₐ, xₚ, xₙ)
    if mod(iter, 20)==0
        xₐᵥ, xₚᵥ, xₙᵥ = SampleTriplets(val..., length(val[2]), false); # There is sampling too
        v_loss = triplet_loss(metric, xₐᵥ, xₚᵥ, xₙᵥ, α); # Just approximation -> correlates with choices of xₐᵥ, xₚᵥ, xₙᵥ 
        v_acc = triplet_accuracy(metric, xₐᵥ, xₚᵥ, xₙᵥ);
        loss_dict = Dict("Training/Loss"=>loss_, "Training/TripletAccuracy"=>acc_,"Validation/Loss"=>v_loss, "Validation/TripletAccuracy"=>v_acc)
        if  β !== 0
            v_reg =  β .* sqrt(sum(x->sqnorm(x,γ), Flux.params(metric)))
            v_par_norm =  sqrt(sum(x->sqnorm(x,0), Flux.params(metric)))
            reg_dict = Dict("Validation/Regularization" =>  v_reg, "Validation/ParamNorm"=>v_par_norm)
            loss_dict = merge(loss_dict, reg_dict)
        end
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


#evaluation SVM and KNN
gm_tr = gram_matrix(train[1], train[1], metric, verbose=false)
gm_val = gram_matrix(train[1], val[1], metric, verbose=false)
gm_tst = gram_matrix(train[1], test[1], metric, verbose=false)


res = []
res_vals = []
for i ∈ 1:10:100
    Γ = 1/i
    # for triplet loss
    model = svmtrain(exp.(Γ .* gm_tr), train[2]; kernel=LIBSVM.Kernel.Precomputed, verbose=true);

    y_valid_pr, _ = svmpredict(model, exp.(Γ .* gm_val));
    y_test_pr, _ = svmpredict(model, exp.(Γ .* gm_tst));

    valid_a = mean(y_valid_pr .== val[2])
    test_a = mean(y_test_pr .== test[2])
    push!(res, ["SVM", Γ, valid_a, test_a])
end
df_svm = DataFrame(permutedims(hcat(res...),(2,1)), ["Model", "Γ/k", "valid_acc", "test_acc"])
df_svm_top = sort(df_svm, ["test_acc"], rev=true)[1,:]

# KNN
val_probs = knn_predict_multiclass(gm_val, train[2])
tst_probs = knn_predict_multiclass(gm_tst, train[2])

tr_len = length(train[2]);
accuracy_val = mean(val_probs .== repeat(val[2], 1, tr_len)', dims=2)[:];
accuracy_tst = mean(tst_probs .== repeat(test[2], 1, tr_len)', dims=2)[:];

k = argmax(accuracy_tst)
push!(res, ["KNN", k, accuracy_val[k], accuracy_tst[k]])


update_config!(lg, Dict("SVM_gamma" => df_svm_top["Γ/k"], "SVM_valid" => df_svm_top["valid_acc"], "SVM_test"=> df_svm_top["test_acc"], "KNN_k"=>k, "KNN_valid"=>accuracy_val[k], "KNN_test"=>accuracy_tst[k]))

#Wandb.log(lg, Wandb.Table(data=permutedims(hcat(res...),(2,1)), columns=["Model", "Γ/k", "valid_acc", "test_acc"]))
# Finish the run (Logger)
close(lg)

df = DataFrame(permutedims(hcat(res...),(2,1)), ["Model", "Γ/k", "valid_acc", "test_acc"])


id = (seed=seed, ui=ui, reg=reg)
savef = joinpath(datadir("triplet", dataset, "$(seed)"), "$(run_name).bson");
results = (
    model=metric, metric=_metric, seed=seed, params=ps, iters=iters, 
    learning_rate=learning_rate, batch_size=batch_size, history=history, 
    train=train, val=val, test=test, ui=id[:ui], res=df, 
)

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"
