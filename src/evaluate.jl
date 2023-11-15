using Statistics, Plots, CSV, BSON, DrWatson, DataFrames, ProgressBars

function summarize_results(folder)
    pth = datadir(folder)
    files = readdir(pth)

    names = ["auc_clf", "auc_ad", "accuracy", "Kg", "2", "6", "KE", "L*", "Kd", "A", "Ke", "E", "Kf", "I", "KU", "*", "K", "Kk", "Kc"]
    value_matrix = []

    for file in tqdm(files)
        loaded = BSON.load(joinpath(pth, file))
        #print(loaded |> keys)
        #println(loaded[:weights])
        push!(value_matrix, [loaded[:auc_clf], loaded[:auc_ad], loaded[:accuracy], values(loaded[:weights])...])
    end
    return DataFrame(value_matrix, names)
end

function summarize_results(folder)
    pth = datadir(folder)
    files = readdir(pth)

    names = ["neighbors", "auc_clf", "accuracy", "Kg", "2", "6", "KE", "L*", "Kd", "A", "Ke", "E", "Kf", "I", "KU", "*", "K", "Kk", "Kc"]
    #["auc_clf", "auc_ad", "accuracy", "Kg", "2", "6", "KE", "L*", "Kd", "A", "Ke", "E", "Kf", "I", "KU", "*", "K", "Kk", "Kc"]
    value_matrix = []
    #tmp = []
    i = 1
    for file in tqdm(files)
        loaded = BSON.load(joinpath(pth, file))
        #print(loaded |> keys)
        #println(loaded[:weights])
        ks = length(loaded[:auc_clf])
        auc_clf = reshape(loaded[:auc_clf], :, 1)
        auc_ad = reshape(loaded[:auc_ad], :, 1)
        acc = reshape(loaded[:accuracy], :, 1)
        k = reshape(collect(1:ks), :, 1)
        weight = repeat(reshape(collect(values(loaded[:weights])), 1, :), ks, 1)
        push!(value_matrix, cat([k, auc_clf, acc, weight]..., dims=2))
        i += 1
        (i==3) ? break : continue
    end
    output = value_matrix[1]
    foreach(x->output = cat([output, x]..., dims=1), tqdm(value_matrix[2:end]))
    #return DataFrame(cat(value_matrix..., dims=1), names)
    return DataFrame(output, names)
end