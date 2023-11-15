t1, t2 = (Mill.ArrayNode(randn(Float32, 13, 1)), Mill.ArrayNode(randn(Float32, 13, 1)))
t3 = Mill.ArrayNode(randn(Float32, 13, 0))


p1 = Mill.ProductNode(
    (
        a1 = Mill.ArrayNode(randn(Float32, 2,7)), 
        a2 = Mill.ArrayNode(randn(Float32, 3,7))
    ))

p2 = Mill.ProductNode(
    (
        a1 = Mill.ArrayNode(randn(Float32, 2,7)), 
        a2 = Mill.ArrayNode(randn(Float32, 3,7))
    ))

p3 = Mill.ProductNode(
    (
        a1 = Mill.ArrayNode(randn(Float32, 2,0)), 
        a2 = Mill.ArrayNode(randn(Float32, 3,0))
    ))


b1 = Mill.BagNode(
    Mill.ProductNode(
        (
            a1 = Mill.ArrayNode(randn(Float32, 2,7)), 
            a2 = Mill.ArrayNode(randn(Float32, 3,7))
        )
    ), 
    Mill.AlignedBags(1:4, 5:5, 6:7)
)

b2 = Mill.BagNode(
    Mill.ProductNode(
        (
            a1 = Mill.ArrayNode(randn(Float32, 2,8)), 
            a2 = Mill.ArrayNode(randn(Float32, 3,8))
        )
    ), 
    Mill.AlignedBags(1:4, 5:8)
)

b3 = Mill.BagNode(
    Mill.ProductNode(
        (
            a1 = Mill.ArrayNode(randn(Float32, 2,0)), 
            a2 = Mill.ArrayNode(randn(Float32, 3,0))
        )
    ), 
    Mill.AlignedBags(0:-1)
)




p1 = Mill.ProductNode(
    (
        a1 = Mill.ArrayNode(randn(Float32, 2,7)), 
        a2 = Mill.ArrayNode(Flux.onehotbatch(rand(1:10, 7), 1:10))
    ))

p2 = Mill.ProductNode(
    (
        a1 = Mill.ArrayNode(randn(Float32, 2,7)), 
        a2 = Mill.ArrayNode(Flux.onehotbatch(rand(1:10, 7), 1:10))
    ))



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



am = LeafMetric(dist_, "a1")

pm = ProductMetric(
           (a1 = LeafMetric(dist_, "a1") , a2 = LeafMetric(dist_, "a2")),
           (p,w) -> dropdims(sqrt.(sum(w .* (p.^2), dims=3)), dims=3),
           (a1 = 10, a2 = 100),
           "pm1"
           )

bm = SetMetric(
    ProductMetric(
        (a1 = LeafMetric(dist_, "a1") , a2 = LeafMetric(dist_, "a2")),
        (p,w) -> dropdims(sqrt.(sum(w .* (p.^2), dims=3)), dims=3),
        (a1 = 10, a2 = 100),
        "pm1"
    ),
    x->CD(x, Flux.mean),
    "bm1"
)

bm1 = SetMetric(
    ProductMetric(
        (a1 = LeafMetric(dist_, "a1") , a2 = LeafMetric(dist_, "a2")),
        WeightedProductMetric,
        (a1 = 10, a2 = 100),
        "pm1"
    ),
    ChamferDistance,
    "bm1"
)



m = _reflectmetric(p1, prod_dist_, CD,  dist_, ones, "")