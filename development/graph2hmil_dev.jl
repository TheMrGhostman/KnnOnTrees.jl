using Revise
using MLDatasets, Mill, Flux, Graphs, StatsBase, GraphRecipes, Plots, Zygote
using HMillDistance, Distances
using OptimalTransport, Tulip, LinearAlgebra

dataset = MLDatasets.TUDataset("MUTAG")


g1 = dataset[1]
g1.graphs

function graph2hmill(gr, un=6, depth::Int=3, pad::bool=false)
    """
    gr .... graph
    un .... number of unique nodes
    """

    # to Directed graph
    N = gr.graphs.num_nodes
    dg = SimpleDiGraph(N)
    for (i,o) in zip(gr.graphs.edge_index...); add_edge!(dg, i, o); end

    # FIXME -> one additional node for empty node placeholder -> (N+1)-th 
    features = Flux.onehotbatch(
        (pad) ? vcat(gr.graphs.node_data.targets, un) : gr.graphs.node_data.targets, 
        0:un
    )
   
    #l1 
    #@info vcat(dg.badjlist, [[0]])
    nodes_idxs = vcat(dg.badjlist, [[N+1]])
    cs = cumsum(length.(nodes_idxs))
    ab, counter = [], 1
    for c in cs; push!(ab, counter:c); counter = c+1; end

    l1_bags = AlignedBags(ab...)#ScatteredBags(nodes_idxs)
    l1_data = features[:,vcat(nodes_idxs...)]

    # l2
    l1 = nodes_idxs
    l2 = [[l1[e] for e in l1[i]] for i ∈ eachindex(l1)]


    return BagNode(l1_data, l1_bags)
    
    #l=1
    #return features
    #BagNode(features, bags_l1)
    BagNode(
        ProductNode(
            (data = ArrayNode(features),
            bonds = BagNode(
                features, 
                l1_bags
            ))),
        ScatteredBags([[i] for i ∈ 1:N]) #TODO think about adding [N+1]
    )   
    
end

out = graph2hmill(g1, 6)
out |> printtree



g1 = dataset[1]
#g1 = dataset[6] # (graphs = Graph(28, 62), target = 1)

un=6
N = g1.graphs.num_nodes
dg = SimpleDiGraph(N)
for (i,o) in zip(g1.graphs.edge_index...); add_edge!(dg, i, o); end
features = Flux.onehotbatch(vcat(g1.graphs.node_data.targets, un), 0:un)


dg

#lvl 0
l0 = AlignedBags([1:N])

# lvl 1
l1 = dg.badjlist #[l1[i] for i ∈ eachindex(l1)]
l1cs = cumsum(length.(l1))
l1_bags = AlignedBags([i:j for (i, j) in zip(vcat(1, l1cs), vcat(l1cs, l1cs[end]+1))])
l1_vec = vcat(l1..., N+1)
l1_data = features[:, l1_vec]


# lvl 2
l2 = [[l1[e] for e in l1[i]] for i ∈ eachindex(l1)]
l2_vec = vcat(vcat(l2...)..., N+1)
l2cs = cumsum(length.(vcat(l2...)))
l2_bags = AlignedBags([i:j for (i, j) in zip(vcat(1, l2cs), vcat(l2cs, l2cs[end]+1))])
l2_data = features[:, l2_vec]

# lvl 3 
l3 = [[[l1[k] for k in l1[e]] for e in l1[i]] for i ∈ eachindex(l1)]
l3_vec = vcat(vcat(vcat(l3...)...)..., N+1)
l3cs = cumsum(length.(vcat(vcat(l3...)...)))
l3_bags = AlignedBags([i:j for (i, j) in zip(vcat(1, l3cs), vcat(l3cs, l3cs[end]+1))])
l3_data = features[:, l3_vec]


# lvl 4
l4 = [[[[l1[d] for d in l1[c]] for c in l1[b]] for b in l1[a]] for a ∈ eachindex(l1)]
l4_vec = vcat(vcat(vcat(vcat(l4...)...)...)..., N+1)
l4cs = cumsum(length.(vcat(vcat(vcat(l4...)...)...)))
l4_bags = AlignedBags([i:j for (i, j) in zip(vcat(1, l4cs), vcat(l4cs, l4cs[end]+1))])
l4_data = features[:, l4_vec]



# Reverse
l4_data
#l4 
r4 = SegmentedSum(7)(l4_data, l4_bags)
#l3
r3 = SegmentedSum(7)(r4, l3_bags)#(l3_data, l3_bags)
#l2
r2 = SegmentedSum(7)(r3, l2_bags)#(l2_data, l2_bags)
#l1
r1 = SegmentedSum(7)(r2, l1_bags)# (l1_data, l1_bags)
#l0 
r0 = SegmentedSum(7)(r1, l0)
