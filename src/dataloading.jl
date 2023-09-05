function get_list_of_datasets()
    datasets = [
        "Mutagenesis", map(x->split(x, ".")[1], readdir(DrWatson.datadir("relational")))...
    ]
    return datasets
end


function load_dataset(name; to_mill=true)
    @assert name in get_list_of_datasets()

    name = lowercase(name)
    if  name == "mutagenesis"
        data = MLDatasets.Mutagenesis(split=:all);
        X = data.features
        y = data.targets
    else 
        data_string = read(datadir("relational/$(name).json"), String);
        data = JSON3.read(data_string);
        X = data.x
        y = data.y
    end
    X = (to_mill) ? _to_mill(X) : X
    return X, y
end


function _to_mill(x)
    sch = JsonGrinder.schema(x);
    extractor = JsonGrinder.suggestextractor(sch);
    # access ngram string as data.S
    return extractor.(x);  
end


function preprocess(X, y; ratios=(0.6,0.2,0.2), procedure=:clf, seed=666)
    @assert sum(ratios) == 1 && length(ratios) == 3
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