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


include("dataloading.jl")
export load_dataset, _to_mill, get_list_of_datasets, preprocess

include("knn_core.jl")
export knn, knn_probs, knn_probs_all, knn_predict_multiclass

include("utils.jl")
export sample_weights, get_most_occured_class

end # module 
