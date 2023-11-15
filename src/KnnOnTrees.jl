module KnnOnTrees

# General
using DrWatson
using Statistics, ProgressBars
using Mill
using GHMill # treeloss function 
# dataloading
using MLDatasets, JSON3, JsonGrinder, MLUtils, Random
# utils 
using Distributions, DataFrames, StatsBase
# treeloss
using Flux, Distances, StringDistances, OneHotArrays, Base.Threads
# stuct
using Zygote

using HMillDistance


include("dataloading.jl")
export load_dataset, _to_mill, get_list_of_datasets, preprocess

include("knn_core.jl")
export knn, knn_tm, knn_probs, knn_probs_all, knn_predict_multiclass

include("utils.jl")
export sample_weights, get_most_occured_class, load_hyperparams

include("treeloss/utils.jl")
export SampleTriplets #zerocardinality, WeightStruct, 

#include("treeloss/metric.jl")
#export LeafMetric, ProductMetric, SetMetric, reflectmetric

#include("treeloss/printing.jl")

include("treeloss.jl")
export weighted_tree_distance

end # module 
