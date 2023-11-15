using Flux, KnnOnTrees, Mill, Random, Distances

tm1 = Mill.ProductNode(
        (
            a1=Mill.ArrayNode(Flux.onehotbatch(rand(1:11, 1), 1:11)), 
            a2=Mill.ArrayNode(randn(Float32, 12, 1)), 
            b2=Mill.BagNode(
                Mill.ProductNode(
                    (
                        a1=Mill.ArrayNode(randn(Float32, 5, 4)), 
                        b4=Mill.BagNode(
                            Mill.ProductNode(
                                (
                                    a1 = Mill.ArrayNode(Flux.onehotbatch(rand(1:2, 7), 1:2)), 
                                    a2 = Mill.ArrayNode(randn(Float32, 3,7))
                                )
                            ), 
                            Mill.AlignedBags(1:1, 2:4, 5:5, 6:7)
                        ), 
                        a2=Mill.ArrayNode(randn(Float32, 6, 4))
                    )
                ), 
                Mill.AlignedBags(1:4)
            ), 
            a3=Mill.ArrayNode(Flux.onehotbatch(rand(1:10, 1), 1:10)), 
            a4=Mill.ArrayNode(randn(Float32, 14, 1))
        )
    )

tm2 = Mill.ProductNode(
    (
        a1=Mill.ArrayNode(Flux.onehotbatch(rand(1:11, 1), 1:11)), 
        a2=Mill.ArrayNode(randn(Float32, 12, 1)), 
        b2=Mill.BagNode(
            Mill.ProductNode(
                (
                    a1=Mill.ArrayNode(randn(Float32, 5, 3)), 
                    b4=Mill.BagNode(
                        Mill.ProductNode(
                            (
                                a1 = Mill.ArrayNode(Flux.onehotbatch(rand(1:2, 8), 1:2)), 
                                a2 = Mill.ArrayNode(randn(Float32, 3,8))
                            )
                        ), 
                        Mill.AlignedBags(1:1, 2:4, 5:8)
                    ), 
                    a2=Mill.ArrayNode(randn(Float32, 6, 3))
                )
            ), 
            Mill.AlignedBags(1:3)
        ), 
        a3=Mill.ArrayNode(Flux.onehotbatch(rand(1:10, 1), 1:10)), 
        a4=Mill.ArrayNode(randn(Float32, 14, 1))
    )
)

t1 = Mill.ProductNode(
        (
            a1=Mill.ArrayNode(randn(Float32, 11, 1)), 
            a2=Mill.ArrayNode(randn(Float32, 12, 1)), 
            b2=Mill.BagNode(
                Mill.ProductNode(
                    (
                        a1=Mill.ArrayNode(randn(Float32, 5, 4)), 
                        b4=Mill.BagNode(
                            Mill.ProductNode(
                                (
                                    a1 = Mill.ArrayNode(randn(Float32, 2,7)), 
                                    a2 = Mill.ArrayNode(randn(Float32, 3,7))
                                )
                            ), 
                            Mill.AlignedBags(1:1, 2:4, 5:5, 6:7)
                        ), 
                        a2=Mill.ArrayNode(randn(Float32, 6, 4))
                    )
                ), 
                Mill.AlignedBags(1:4)
            ), 
            a3=Mill.ArrayNode(randn(Float32, 13, 1)), 
            a4=Mill.ArrayNode(randn(Float32, 14, 1))
        )
    )

t2 = Mill.ProductNode(
    (
        a1=Mill.ArrayNode(randn(Float32, 11, 1)), 
        a2=Mill.ArrayNode(randn(Float32, 12, 1)), 
        b2=Mill.BagNode(
            Mill.ProductNode(
                (
                    a1=Mill.ArrayNode(randn(Float32, 5, 3)), 
                    b4=Mill.BagNode(
                        Mill.ProductNode(
                            (
                                a1 = Mill.ArrayNode(randn(Float32, 2,7)), 
                                a2 = Mill.ArrayNode(randn(Float32, 3,7))
                            )
                        ), 
                        Mill.AlignedBags(1:4, 5:5, 6:7)
                    ), 
                    a2=Mill.ArrayNode(randn(Float32, 6, 3))
                )
            ), 
            Mill.AlignedBags(1:3)
        ), 
        a3=Mill.ArrayNode(randn(Float32, 13, 1)), 
        a4=Mill.ArrayNode(randn(Float32, 14, 1))
    )
)



t1 |> printtree
metric = reflectmetric(t1);
metric |> printtree

metric(t1, t2)
metric(t2, t1)

ps = Flux.params(metric)

grad = gradient(()->Flux.mean(metric(t1,t2)), ps)

map(k->grad[k], keys(grad))

#TODO figure out why Cityblock matric does not work 



dummy_encoder = GHMill.reflectinencoder(t1; verbose=false);
structure, W_init = GHMill.tree_struct_and_weights_dict(dummy_encoder; rwi=true);
ws = WeightStruct(W_init);

weighted_tree_distance(dummy_encoder(t1)[1], dummy_encoder(t2)[1], ws, structure["*"], "*"; just_in=true )