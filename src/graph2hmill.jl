"""
    graph2hmill(graph::MLDatasets.Graph, unique_nodes::Int, depth::Int=3; pad::Bool=true)

    Function takes MLDatasets.Graph and transform it into Hierarchical Multiinstance Learning data 
    (Mill.jl framework), while new HMIL data are formated to be equivalent to procedure described in paper 
    
    Ching-Yao Chuang and Stefanie Jegelka (2022)
        Tree Mover's Distance: Bridging Graph Metrics and Stability of Graph Neural Networks

}
"""
function graph2hmill(graph::MLDatasets.Graph, unique_nodes::Int, depth::Int=3; pad::Bool=true)
    # to Directed graph Type -> just simplifies extraction of badjlist etc. (+ ploting is available etc.)
    @assert depth > 0
    N = graph.num_nodes
    dg = SimpleDiGraph(N)
    for (i,o) in zip(graph.edge_index...); add_edge!(dg, i, o); end
 
    features = Flux.onehotbatch(graph.node_data.targets, 0:unique_nodes-1)
    features = (pad) ? hcat(features, zeros(eltype(features), unique_nodes, 1)) : features # if to pad
   
    # if pad => UN x (N+1) else => UN x N
    # lvl 0 -> "root" to all vertices (nodes)
    lvl0 = AlignedBags([1:N]) # always N not (N+1)
    lvl1 = dg.badjlist # lists of nodes that are directed to l0
    if depth == 1
        BagNode(features, lvl0)
    else
        BagNode(recursive_levels(depth-1, lvl1, features, lvl1, N, pad), lvl0)
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
                bonds=recursive_levels(lvl_to_go-1, new_nodes, features, edges, n)
                )), 
            bags
        )
    end
end

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