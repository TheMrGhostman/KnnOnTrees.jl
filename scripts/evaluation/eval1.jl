using DataFrames, BSON, DrWatson
using KnnOnTrees, Mill, LinearAlgebra, Statistics, Flux
using Plots, StatsPlots
using Plots.PlotMeasures

# Default values
dataset = "Mutagenesis"

df = DrWatson.collect_results(datadir("triplet", dataset), subfolders=true);
df |> names

par_name= [
    "atoms.bond.element", "atoms.bond.bond_type", "atoms.bond.charge", "atoms.bond.atom_type", 
    "atoms.element", "atoms.bonds", "atoms.charge", "atoms.atom_type",
    "lumo", "inda", "logp", "ind1", "atoms"
]

par_names= [
    "a.b.element (cat)", "a.b.bond_type (cat)", "a.b.charge (con)", "a.b.atom_type (cat)", 
    "a.element (cat)", "a.bonds (set)", "a.charge (con)", "a.atom_type (cat)",
    "lumo (con)", "inda (cat)", "logp (con)", "ind1 (cat)", "atoms (set)"
]



gr = df[:,["batch_size", "ui"]];
argidx= sortperm(gr[:,"ui"]);


pr = map(x->softplus.(vcat((x.params |> collect)...)), df[:,"params"]);
pr_mat= hcat(pr...);

pr3d = reshape(pr_mat[:,argidx], (13,3,6));


p = plot(layout=(6,1), size=(1000,1200), left_margin = [20mm 0mm], xrotation=-45); #axis=([], false), 
for i in 1:6
    if i!=6
        p = boxplot!(p, repeat(1:13,outer=3), reshape(pr3d[:,:,i], :), subplot=i, xformatter=_->"", legend=false);
    else
        p = boxplot!(p, repeat(1:13,outer=3), reshape(pr3d[:,:,i], :), subplot=i, xticks = (1:13, par_names), 
        bottom_margin = 20mm, legend=false);
    end
end
savefig("boxplots");

"""
p = plot(layout=(6,1), size=(1000,1200), left_margin = [25mm 0mm], xrotation=-45); #axis=([], false), 
for i in 1:6
    if i!=6
        p = boxplot!(p, repeat(1:13,outer=3), log10.(reshape(pr3d[:,:,i], :)), subplot=i, ylabel="10^y",
        yguidefontrotation=-90, xformatter=_->"", legend=false);
    else
        p = boxplot!(p, repeat(1:13,outer=3), log10.(reshape(pr3d[:,:,i], :)), subplot=i, ylabel="10^y",
        yguidefontrotation=-90, xticks = (1:13, par_names), bottom_margin = 20mm, legend=false);
    end
end
"""


p = plot(layout=(6,1), size=(1000,1200), left_margin = [10mm 0mm], xrotation=-45) #axis=([], false), 
for i in 1:6
    if i!=6
        p = boxplot!(p, repeat(1:13,outer=3), reshape(pr3d[:,:,i], :), subplot=i, yaxis=:log,
        xformatter=_->"", legend=false)
    else
        p = boxplot!(p, repeat(1:13,outer=3), reshape(pr3d[:,:,i], :), subplot=i, yaxis=:log,
        xticks = (1:13, par_names), bottom_margin = 20mm, legend=false)
    end
end
savefig("boxplots-log");




