function block_segmented_norm(x::AbstractMatrix{<:Real}, seg1, seg2, ::typeof(sum))
    forward_block_segmented_sum(x, seg1 , seg2)
end

function block_segmented_norm2(x::AbstractMatrix{<:Real}, seg1, seg2, norm::Function=sum)
    [norm(x[sᵢ, sⱼ]) for sᵢ in seg1, sⱼ in seg2]
end

function forward_block_segmented_sum(x, seg1 , seg2)
    o = similar(x, length(seg1), length(seg2))
    for (j, sⱼ) in enumerate(seg2)
        for (i, sᵢ) in enumerate(seg1)
            r = zero(eltype(x))
            @inbounds for l in sⱼ
                for k in sᵢ
                    r += x[k,l]
                end
            end
            o[i,j] = r
        end
    end
    o
end

function reverse_block_segmented_sum(ȳ, x, seg1 , seg2)
    o = zero(x)
    for (j, sⱼ) in enumerate(seg2)
        for (i, sᵢ) in enumerate(seg1)
            r = zero(eltype(x))
            @inbounds for l in sⱼ
                for k in sᵢ
                    o[k,l] += ȳ[i,j]

                end
            end
        end
    end
    (NoTangent(), o, NoTangent(), NoTangent())
end

function ChainRulesCore.rrule(::typeof(block_segmented_norm), x::AbstractMatrix{<:Real}, seg1, seg2, ::typeof(sum))
    o = forward_block_segmented_sum(x, seg1, seg2)
    o, ȳ -> reverse_block_segmented_sum(ȳ, x, seg1 , seg2)

end