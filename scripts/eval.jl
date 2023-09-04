using Statistics, Plots, CSV, BSON, DrWatson, DataFrames, ProgressBars
using Plots.PlotMeasures



#["neighbors", "auc_clf", "accuracy", "Kg", "2", "6", "KE", "L*", "Kd", "A", "Ke", "E", "Kf", "I", "KU", "*", "K", "Kk", "Kc"]
dataframe = CSV.read("/home/zorekmat/MIL/GHMill.jl/experiments/knn_treeloss/data/overall_rs_table.csv", DataFrame);

#gbk = groupby(df, :neighbors);

get_weight_names() = ["Kg", "2", "6", "KE", "L*", "Kd", "A", "Ke", "E", "Kf", "I", "KU", "*", "K", "Kk", "Kc"]

function marginal_plot(df, groupby_=:neighbors, key=:auc_clf)
    gbk = groupby(df, groupby_);
    
    means = combine(gbk, key => mean)[:, Symbol(String(key)*"_mean")];
    stds = combine(gbk, key => std)[:, Symbol(String(key)*"_std")];
    maxs = combine(gbk, key => maximum)[:,  Symbol(String(key)*"_maximum")];
    mins = combine(gbk, key => minimum)[:,  Symbol(String(key)*"_minimum")];
    xaxis = map(x->gbk[x][1,groupby_], 1:length(gbk))
    order_ = sortperm(xaxis)
    
    plot(xaxis[order_], means[order_] .- 2 .* stds[order_], fillrange = means[order_] .+ 2 .* stds[order_], fillalpha=0.35, c = 1, label="μ ± 2⋅σ", bottom_margin=10mm, left_margin=10mm, xlabel="k", ylabel="AUC");
    savefig("figures/plots/marginals--$(String(key))--$(String(groupby_))--CI");

    plot(xaxis[order_], means[order_] .- 2 .* stds[order_], fillrange = means[order_] .+ 2 .* stds[order_], fillalpha=0.35, c = 1, label="μ ± 2⋅σ", bottom_margin=10mm, left_margin=10mm, xlabel="k", ylabel="AUC");
    plot!(xaxis[order_], means[order_], line=:scatter, c=2, msw = 0, ms=2.5, label = "μ");
    plot!(xaxis[order_], maxs[order_], line=:scatter, c=3, msw = 0, ms=2.5, label = "max");
    plot!(xaxis[order_], mins[order_], line=:scatter, c=4, msw = 0, ms=2.5, label = "min");
    #plot!(label = "argmax_k = $(findmax(maxs))");
    savefig("figures/plots/marginals--$(String(key))--$(String(groupby_))");
end
# 

function find_max(df, key=:auc_clf, topk=10)
    sdf= sort(df, :auc_clf, rev = true)
    return sdf[1:topk, :]
end

function marginal_histograms(df)
    for i in ["neighbors", "Kg", "2", "6", "KE", "L*", "Kd", "A", "Ke", "E", "Kf", "I", "KU", "*", "K", "Kk", "Kc"]
        histogram(df[:,i], bottom_margin=10mm, left_margin=10mm,xlabel=Symbol(i)); # 
        savefig("figures/histograms/histogram--$(String(i))")
    end
end

top1000 = find_max(dataframe, :auc_clf, 1000);
top1000[:, get_weight_names()] |> unique # unique weight combinations in top1000
