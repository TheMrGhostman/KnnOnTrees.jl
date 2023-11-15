using Mill, Random

function get_dummy_trees()
    Random.seed!(666)
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
    return t1,t2
end


get_array_nodes() = (Mill.ArrayNode(randn(Float32, 13, 1)), Mill.ArrayNode(randn(Float32, 13, 1))) 