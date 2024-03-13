"""
    BalancedDisjunctBinaryBatches(labels, batch_size; seed=Int(rand(1:1e5)))

Function creates batches from data with binary class labels. 
Every batch contain samples from both classes in the same ration as are classes in whole dataset.
Batches are furthermore disjunct and fixed!!! Both can be changed later. Idea behind these two decisions is 
then cache can be later joint together and used in posterior extimation.
#TODO add unfixed and not disjunct version.
"""
function BalancedDisjunctBinaryBatches(labels, batch_size; seed=Int(rand(1:1e5)))
    nb = round(length(labels) / batch_size)
    cls0 = findall(labels .== 0); 
    cls1 = findall(labels .== 1);
    c0perb = round(length(cls0) / nb)
    c1perb = round(length(cls1) / nb)
    Random.seed!(seed)
    cls0 = chunk(cls0[randperm(length(cls0))], Int(c0perb))
    cls1 = chunk(cls1[randperm(length(cls1))], Int(c1perb))
    batches = [vcat(c0,c1) for (c0, c1) ∈ zip(cls0, cls1)]
    batches = [batch[randperm(l)] for (batch, l) in zip(batches, length.(batches))]
    return batches
end


function BalancedDisjunctBatches(labels, batch_size, n_classes; seed=Int(rand(1:1e5)))
    nb = round(length(labels) / batch_size)
    cls = Dict([i => findall(labels .== i) for i in 1:n_classes]); 
    cperb = Dict([i => round(length(cls[i]) / nb) for i in keys(cls)]); 
    Random.seed!(seed)
    cls0 = [chunk(cls[i][randperm(length(cls[i]))], Int(cperb[i])) for i in keys(cls)]
    common_denominator = minimum(getindex.(size.(cls0), 1))
    batches = [vcat([cls0[j][i] for j ∈ 1:n_classes]...) for i ∈ 1:common_denominator]
    classes = [vcat([j .* ones(size(cls0[j][i])) for j ∈ 1:n_classes]...) for i ∈ 1:common_denominator]
    return classes, classes
end

chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]

function MOLabels(y, out_dim)
    # repeat(y, out_dim) ... 
    @assert issubset(unique(y), 1:out_dim+1)
    new_y = vcat([float.(y .== i) for i ∈ 1:out_dim]...)
    return new_y
end
