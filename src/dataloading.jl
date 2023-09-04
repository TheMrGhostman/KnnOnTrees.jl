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