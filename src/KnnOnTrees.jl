module KnnOnTrees

# General
using DrWatson
using Statistics, ProgressBars
using Mill
using GHMill # treeloss function 
# dataloading
using MLDatasets, JSON3, JsonGrinder


include("dataloading.jl")
export load_dataset, _to_mill, get_list_of_datasets

include("core.jl")
export knn, knn_probs, knn_probs_all

end # module knn_treeloss
