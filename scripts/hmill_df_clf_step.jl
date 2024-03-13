using ArgParse, DrWatson
using BSON, StatsBase, Statistics
using MLDatasets, MLUtils, Mill, JsonGrinder, JSON3, OneHotArrays, Flux
#using GHMill
using KnnOnTrees, ProgressBars, DataFrames, CSV
using Wandb, Dates, Logging, ProgressBars

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        arg_type = String
        help = "Name of dataset to use"
    "table_idx"
        arg_type = Int
        help = "index in table of all possible samples"
    "seed"
        arg_type = Int
        help = "Random seed for initialization of data splits"
        default = 666
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, table_idx, seed = parsed_args
# dataset, table_idx, seed = "Mutagenesis", 1, 666

tab_path = datadir("rs_options", "hmill_rs.csv") # $(dataset)
df = CSV.read(tab_path, DataFrame);

params_ = load_hyperparams(df, table_idx)
params_ = merge(params_, (;patience=30,))

run_name = "HMillClassifier-$(dataset)-seed=$(seed)-ui=$(table_idx)"
# Initialize logger
lg = WandbLogger(project ="TripletLoss",
                 name = run_name,
                 config = Dict{String, Any}([String.(sym)=>val for (sym,val) in pairs(params_)]))

update_config!(lg, Dict("ui"=>table_idx,))

global_logger(lg)
# 1) sample parameters
#epochs=1000
#hdim = 10
#agg = SegmentedMeanMax
#bs = 32
#lr = 1e-3
#patience=10
agg = if params_[:agg] == "SegmentedMeanMax"
    SegmentedMeanMax
elseif params_[:agg] == "SegmentedMax"
    SegmentedMax
elseif params_[:agg] == "SegmentedMean"
    SegmentedMean
elseif params_[:agg] == "SegmentedSum"
    SegmentedSum
else 
    @error "unkown aggregation"
end

data = load_dataset(dataset; to_mill=true);
data[2] .= binary_class_transform(data[2], (1,2))
train, val, test = preprocess(data...; ratios=(0.6,0.2,0.2), procedure=:clf, seed=seed);


# initialize model
classes = length(unique(data[2]));
clf_head = Dense(params_[:hdim], classes);
model = clf_head ∘ Mill.reflectinmodel(data[1][1], d->Dense(d, params_[:hdim]), agg);

ps = Flux.params(model);
opt = AdaBelief(params_[:lr]);

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), onehotbatch(y, 1:classes))
accuracy(x, y) = mean(Flux.onecold(model(x)) .== y)

data_loader = Flux.DataLoader(train, batchsize=params_[:bs], shuffle=true);

history = Dict("epochs"=>[], 
    "train_loss" => [], "test_loss" => [], "val_loss" => [], 
    "train_acc"=>[], "test_acc"=>[], "val_acc"=>[]
    );
params_ = merge(params_, (patience = 50, ))

global best_model = deepcopy(model);
global criterion = 0
global patience_ = deepcopy(params_[:patience])

start = time()
for i in tqdm(1:params_[:epochs])
    Flux.Optimise.train!(loss, ps, data_loader, opt);

    train_loss, val_loss, test_loss = loss(train[1], train[2]), loss(val[1], val[2]), loss(test[1], test[2])
    train_acc, val_acc, test_acc = accuracy(train[1], train[2]), accuracy(val[1], val[2]), accuracy(test[1], test[2])


    push!(history["epochs"], i)
    push!(history["train_loss"], train_loss)
    push!(history["val_loss"], val_loss)
    push!(history["test_loss"], test_loss)
    push!(history["train_acc"], train_acc)
    push!(history["val_acc"], val_acc)
    push!(history["test_acc"], test_acc)
    Wandb.log(lg, Dict(
        "Training/Loss" => train_loss,
        "Training/Accuracy" => train_acc,
        "Validation/Loss" => val_loss,
        "Validation/Accuracy" => val_acc,
        ))
    if val_acc >= criterion
        global criterion = val_acc
        global best_model = deepcopy(model)
        global patience_ = deepcopy(params_[:patience]) 
    else
        global patience_ -= 1
    end
    if patience_ == 0
        break
    end
end
end_ = time()
#####

y_pred_train = best_model(train[1])
y_pred_val = best_model(val[1])
y_pred_test = best_model(test[1]) 


update_config!(lg, Dict(
    "HMill-val"=>mean(Flux.onecold(y_pred_val) .== val[2]), 
    "HMill-test"=>mean(Flux.onecold(y_pred_test) .== test[2])))

close(lg)

id = merge(params_, (seed=seed, ui=table_idx))
savefm = joinpath(datadir("hmill_rs", "clf", dataset, "$(seed)"), savename("model_hmill", id, "bson", digits=5));
smodel = (
    model=model,
    params=id,
    train_time = end_ - start, 
)
smodel = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(smodel)]); # this has to be a Dict 
tagsave(savefm, smodel, safe = true);

savef = joinpath(datadir("hmill_rs", "clf", dataset, "$(seed)"), savename("hmill", id, "bson", digits=5));

results = (
    y_pred_train = y_pred_train,
    y_pred_val = y_pred_val,
    y_pred_test = y_pred_test,
    y_train = train[2], 
    y_val = val[2],
    y_tst = test[2],
    accuracy_val = mean(Flux.onecold(y_pred_val) .== val[2]),
    accuracy_tst = mean(Flux.onecold(y_pred_test) .== test[2]),
    seed=seed,
    dataset=dataset,
    params=id,
    train_time = end_ - start,
    ui=id[:ui]
);

result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(results)]); # this has to be a Dict 
tagsave(savef, result, safe = true);
@info "Results were saved into file $(savef)"