"""
    graph2hmill(graph::MLDatasets.Graph, unique_nodes::Int, depth::Int=3; pad::Bool=true, tt::Function=identity)

    Function takes MLDatasets.Graph and transform it into Hierarchical Multiinstance Learning data 
    (Mill.jl framework), while new HMIL data are formated to be equivalent to procedure described in paper 
    
    Ching-Yao Chuang and Stefanie Jegelka (2022)
        Tree Mover's Distance: Bridging Graph Metrics and Stability of Graph Neural Networks

    If targets are unresonable, i.e. [5, 10, 100], targets should be transfromed to [0,1,2]. 
    To remove redundant dimensions from OneHot representation, because they do not have any underlaying information.
    (Or we did not find any in their description) \"tt ≈ target transformation\"
}
"""
function graph2hmill(graph::MLDatasets.Graph, unique_nodes::Int, depth::Int=3; pad::Bool=true, tt::Function=identity)
    # to Directed graph Type -> just simplifies extraction of badjlist etc. (+ ploting is available etc.)
    @assert depth > 0
    N = graph.num_nodes
    dg = SimpleDiGraph(N)
    for (i,o) in zip(graph.edge_index...); add_edge!(dg, i, o); end
 
    targets = Flux.onehotbatch(tt.(graph.node_data.targets), 0:unique_nodes-1)
    features = (:features in keys(graph.node_data)) ? vcat(graph.node_data.features, targets) : targets
    dim_feat = (:features in keys(graph.node_data)) ? size(features, 1) : unique_nodes
    features = (pad) ? hcat(features, zeros(eltype(features), dim_feat, 1)) : features # if to pad
   
    # if pad => UN x (N+1) else => UN x N
    # lvl 0 -> "root" to all vertices (nodes)
    lvl0 = AlignedBags([1:N]) # always N not (N+1)
    lvl1 = dg.badjlist # lists of nodes that are directed to l0
    if depth == 1
        BagNode(features, lvl0)
    else
        BagNode(
            ProductNode((
                data = features,
                bonds=recursive_levels(depth-1, lvl1, features, lvl1, N, pad)
                )), 
            lvl0
        )
    end
end

function recursive_levels(lvl_to_go, nodelist, features, edges, n, pad::Bool=true)

    nodes, bags, data_indexes = get_bags_and_indexes(nodelist, n, pad)

    if lvl_to_go == 1
        return BagNode(features[:, data_indexes], bags)
    else
        new_nodes = [[edges[e] for e ∈ nodes[i]] for i ∈ eachindex(nodes)]
        return BagNode(
            ProductNode((
                data=features[:, data_indexes], 
                bonds=recursive_levels(lvl_to_go-1, new_nodes, features, edges, n, pad)
                )), 
            bags
        )
    end
end

"""
    _create_transition_sheet(x::Vector)
"""
function _create_transition_sheet(x::Vector, min=0)
    present_ = sort(unique(x))
    new_ = collect(min:min+length(present_))
    transition_dict = Dict([i => j for (i,j) in zip(present_, new_)])
    return x->transition_dict[x], transition_dict
end


#=
function recursive_levels(lvl_to_go, nodelist, features, edges, n, pad::Bool=true)

    nodes, bags, data_indexes = get_bags_and_indexes(nodelist, n, pad)

    if lvl_to_go == 1
        return BagNode(features[:, data_indexes], bags)
    else
        new_nodes = [[edges[e] for e ∈ nodes[i]] for i ∈ eachindex(nodes)]
        return BagNode(
            ProductNode((
                data=features[:, data_indexes], 
                bonds=recursive_levels(lvl_to_go-1, new_nodes, features, edges, n)
                )), 
            bags
        )
    end
end
=#


function get_bags_and_indexes(list, n, pad::Bool=true)
    if eltype(list) <: Vector{Int}
        cs_list = cumsum(length.(list))
        lower_bounds = vcat(1, cs_list)
        upper_bounds = (pad) ? vcat(cs_list, cs_list[end]+1) : cs_list # expand for last padded last column
        new_bags = AlignedBags([i:j for (i,j) ∈ zip(lower_bounds, upper_bounds)])
        new_data_indexes = (pad) ? vcat(list..., n+1) : vcat(list...)
        return list, new_bags, new_data_indexes
    else
        new_list = vcat(list...)
        return get_bags_and_indexes(new_list, n, pad)
    end
end


function get_n_unique_nodes(dataset)
    x = vcat([dataset[i].graphs.node_data.targets |> unique for i in 1:length(dataset)]...) |> unique
    return x
end

"""
Returns two simple graphs for debugging purposes

    4 ---- 1            1
    |    / |          / |
    |   /  |         /  | 
    |  /   |        /   |
    | /    |       /    |
    3 ---- 2      3 --- 2

"""
function get_dummy_graphs()
    g1 = MLDatasets.Graph(
        4,
        10,
        ([1,1,1, 2,2, 3,3,3, 4,4], [2,3,4, 1,3, 2,1,4, 1,3]),
        (targets=[0,1,2,3], ),
        ()
    )

    g2 = MLDatasets.Graph(
        3,
        6,
        ([1,1, 2,2, 3,3], [2,3, 1,3, 2,1, 1,3]),
        (targets=[0,1,2], ),
        ()
    )
   
    return g1, g2
end