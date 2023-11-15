import HierarchicalUtils: NodeType, LeafNode, InnerNode, children

NodeType(::Type{<:LeafMetric}) = LeafNode()
NodeType(::Type{<:AbstractMetric}) = InnerNode()

children(n::SetMetric) = (n.im,)
children(n::ProductMetric) = n.ms

Base.show(io::IO, m::ProductMetric) = print(io, nameof(typeof(m)), " ↦  $(nameof(m.pm)), (\"$(m.keyname)\"), ", "Weights = ", m.weights)
Base.show(io::IO, m::SetMetric) = print(io, nameof(typeof(m)), " ↦  $(nameof(m.sm)), (\"$(m.keyname)\") ")
Base.show(io::IO, m::LeafMetric) = print(io, nameof(typeof(m)), " ↦  $(nameof(m.metric)), (\"$(m.keyname)\"),  ", "Type = ", m.type)