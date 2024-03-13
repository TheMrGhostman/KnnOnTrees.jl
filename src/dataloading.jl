function get_list_of_datasets()
    datasets = [
        "Mutagenesis", map(x->split(x, ".")[1], readdir(DrWatson.datadir("relational")))...
    ]
    return datasets
end


function load_dataset(name; to_mill::Bool=true, to_pad_leafs::Bool=false, depth::Int=4)
    #@assert name in get_list_of_datasets()
    #name = lowercase(name)
    if  name in ["mutagenesis", "Mutagenesis"]
        data = MLDatasets.Mutagenesis(split=:all);
        X = data.features
        y = data.targets .+ 1
    elseif name in get_list_of_datasets()
        data_string = read(datadir("relational/$(name).json"), String);
        data = JSON3.read(data_string);
        X = data.x
        y = data.y
    else
        data = MLDatasets.TUDataset(name)
        un_classes = get_n_unique_nodes(data)
        n_class =  un_classes |> length
        transf = (n_class < maximum(un_classes)) ? _create_transition_sheet(un_classes, 0)[1] : identity
        transform = (to_mill) ? x->graph2hmill(x, n_class, depth; pad=to_pad_leafs, tt=transf) : identity
        X = [transform(data[i].graphs) for i in only(axes(data))]
        y = data.graph_data.targets
        to_mill = false # no need to du _to_mill anymore
    end
    X = (to_mill) ? _to_mill(X) : X
    return X, Array(y)
end


function _to_mill(x)
    sch = JsonGrinder.schema(x);
    extractor = JsonGrinder.suggestextractor(sch);
    # access ngram string as data.S
    return extractor.(x);  
end


function binary_class_transform(y, new_class_values::Tuple{Number, Number}=(-1,1))
    orig_class_idx = sort(unique(y))
    @assert length(orig_class_idx) == 2
    new_class_idx = sort([new_class_values...])
    if orig_class_idx == new_class_idx
        return y
    else
        new_y = similar(y)
        new_y[y .== orig_class_idx[1]] .= new_class_values[1]
        new_y[y .== orig_class_idx[2]] .= new_class_values[2]
        return new_y
    end
end

function filter_out_classes_under_n_observations(y::AbstractVector, n::Int)
    cm = countmap(y)
    kv = [(k,cm[k]) for k in keys(cm)]
    kept_kv = filter(x->x[2]>n, kv)
    kept_classes = map(x->x[1], kept_kv)
    kept_obs_idx = map(x->(x[2] in kept_classes) ? x[1] : nothing, enumerate(y))
    indices = filter(x->x!==nothing, kept_obs_idx)
    return indices
end


function preprocess(X, y; ratios=(0.6,0.2,0.2), procedure=:clf, seed=666, filter_under=0)
    @assert sum(ratios) == 1 && length(ratios) == 3

    idx = filter_out_classes_under_n_observations(y, filter_under);
    X, y = X[idx], y[idx];
    val_tst_ratio = ratios[2] + ratios[3]
    if procedure==:clf
        Random.seed!(seed)
        (X_tr, y_tr), rest = MLUtils.splitobs((X,y); at=ratios[1], shuffle=true)
        (X_val, y_val), (X_tst, y_tst) = MLUtils.splitobs(rest; at=ratios[2]/val_tst_ratio, shuffle=false)
        Random.seed!()
    elseif procedure==:ad
        Random.seed!(seed)
        @assert unique(y) == 2 "There is multiple classes!! For AD we need only two classes \"0\" and \"1\"."
        # first split clean train/val/test
        (X_tr, y_tr), rest = MLUtils.splitobs((X[y.==0], y[y.==0]); at=ratios[1], shuffle=true)
        (X_cval, y_cval), (X_ctst, y_ctst) = MLUtils.splitobs(rest; at=ratios[2]/val_tst_ratio, shuffle=false)
        # secondly split anomalous to val/test because they are not needed in train 
        (X_aval, y_aval), (X_atst, y_atst) = MLUtils.splitobs((X[y.==1],y[y.==1]); at=0.5, shuffle=true)
        Random.seed!()
        (X_val, y_val) = (vcat(X_cval, X_aval), vcat(y_cval, y_aval))
        (X_tst, y_tst) = (vcat(X_ctst, X_atst), vcat(y_ctst, y_atst)) 
    else
        @error("Unknown preprocessing procedure. Options are :clf for classification and :ad for anomaly detection")
    end
    return (X_tr, y_tr), (X_val, y_val), (X_tst, y_tst)
end