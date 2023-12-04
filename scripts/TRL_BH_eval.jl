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
    "triplet_creation"
        arg_type = String
        help = "Type of creation of triplets -> (\"batch_hard\", \"balanced\", \"switching\")"
        default="batch_hard"
end

parsed_args = parse_args(ARGS, s)
@unpack dataset, seed, iters, learning_rate, batch_size, reg, gamma, margin, ui, log_pars, triplet_creation = parsed_args
# dataset, seed, iters, learning_rate, batch_size, ui = "Mutagenesis", 666, 1000, 1e-2, 10, 111


run_name = "TripletLoss-$(dataset)-seed=$(seed)-ui=$(ui)-TC=$(triplet_creation)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",#"Julia-testing",
                 name = run_name,
                 config = Dict("learning_rate" => learning_rate,
                               "batch_size" => batch_size,
                               "triplet_creation" => triplet_creation,
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
    xₐ, xₚ, xₙ = TripletCreation(triplet_creation, train, batch_size, metric)
    
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
gm_tr = gram_matrix(train[1], train[1], metric, verbose=false);
gm_val = gram_matrix(train[1], val[1], metric, verbose=false);
gm_tst = gram_matrix(train[1], test[1], metric, verbose=false);

rbfkernel(x, γ) = exp.(-(x .^2) ./ γ)


res = []
for γ ∈ tqdm(0:0.01:2)
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
update_config!(lg, Dict("SVM-(γ|t|v|t)" => round.(svm_matrix[argmax_, :], digits=3)))


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
update_config!(lg, Dict("KNN-(k|v|t)" => round.(knn_matrix[argmax_, :], digits=3)))

# Finish the run (Logger)
close(lg)


id = (seed=seed, ui=ui, reg=reg)
savef = joinpath(datadir("triplet", dataset, "$(seed)"), "$(run_name).bson");
results = (
    model=metric, metric=_metric, seed=seed, params=ps, iters=iters, 
    learning_rate=learning_rate, batch_size=batch_size, history=history, 
    train=train, val=val, test=test, ui=id[:ui], 
    svm_res = DataFrame(svm_matrix, svm_columns),
    knn_res = DataFrame(knn_matrix, knn_columns)

)

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"
et = floor(time()-start)
@info "Elapsed time: $(et) s"
