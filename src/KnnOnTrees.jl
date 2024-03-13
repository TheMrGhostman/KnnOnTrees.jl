module KnnOnTrees

# General
using DrWatson
using Statistics, ProgressBars
using Mill
#using GHMill # treeloss function 
# dataloading
using MLDatasets, JSON3, JsonGrinder, MLUtils, Random
# utils 
using Distributions, DataFrames, StatsBase, LinearAlgebra
# treeloss
using Flux, Distances, StringDistances, OneHotArrays, Base.Threads
# stuct
using Zygote
# graph2hmill
using Graphs, GraphRecipes
# GP
using KernelFunctions

#hmill dist
using HMillDistance

#logging
using Wandb


include("dataloading.jl")
export load_dataset, _to_mill, get_list_of_datasets, preprocess, binary_class_transform, filter_out_classes_under_n_observations

include("knn_core.jl")
export knn, knn_tm, knn_probs, knn_probs_all, knn_predict_multiclass, gram_matrix

include("utils.jl")
export sample_weights, get_most_occured_class, load_hyperparams

include("TripletLoss/triplet_creation.jl")
export SampleTriplets, OfflineBatchHardTriplets, TripletCreation#, triplet_loss

include("GaussianProcess/kernels.jl")
export AbstractHMillKernel, LaplacianHMillKernel, Matern32HMillKernel, GaussianHMillKernel, KernelSelector

include("GaussianProcess/gp_batching.jl")
export BalancedDisjunctBinaryBatches, BalancedDisjunctBatches, MOLabels, chunk

include("GaussianProcess/gp_utils.jl")

include("graph2hmill.jl")
export graph2hmill, _create_transition_sheet
#include("treeloss/metric.jl")
#export LeafMetric, ProductMetric, SetMetric, reflectmetric

#include("treeloss/printing.jl")

#include("treeloss.jl")
#export weighted_tree_distance

end # module 
