module KnnOnTrees

using GHMill, Statistics, ProgressBars

include("helper_tools.jl")
export identify_w

include("core.jl")
export knn, knn_probs, knn_probs_all

end # module knn_treeloss
