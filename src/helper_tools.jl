# Helper functions -> needs to be simplified
using GHMill

function identify_w(m::ProductEncoder, tree::Dict=Dict(), weights::Dict=Dict(); verbose=true)
    if verbose
        @info "Type_of_node: ProductNode | name: $(m.name) | leaves: $(m.ms)"
    end
    tree[m.name] = ("ProductNode", Dict())
    weights[m.name] = 1f0
    for ms in m.ms
        identify_w(ms, tree[m.name][2], weights, verbose=verbose)
    end
    return tree
end

function identify_w(m::ArrayEncoder, tree::Dict=Dict(), weights::Dict=Dict(); verbose=true)
    if verbose
        @info "Type_of_node: ArrayNode | name: $(m.name)"
    end
    tree[m.name] = ("ArrayNode",)
    weights[m.name] = 1f0
    return tree
end

function identify_w(m::BagEncoder, tree::Dict=Dict(), weights::Dict=Dict(); verbose=true)
    if verbose
        @info "Type_of_node: BagNode | name: $(m.name) | leaves: $(m.im)"
    end
    tree[m.name] = ("BagNode", Dict())
    weights[m.name] = 1f0
    identify_w(m.im, tree[m.name][2], weights, verbose=verbose)
    return tree
end