"""
# Hypergraph Encoding Engine

This module implements hypergraph encoding of agentic kernel patterns
and cognitive relationships using graph-based representations.
"""

using Graphs

"""
Represents a hypergraph node with cognitive properties.
"""
struct HypergraphNode
    id::Int
    node_type::Symbol  # :kernel, :percept, :action, :memory
    properties::Dict{Symbol, Any}
    activation_level::Float64
    
    function HypergraphNode(id::Int, node_type::Symbol; 
                           properties::Dict{Symbol, Any} = Dict{Symbol, Any}(),
                           activation_level::Float64 = 0.0)
        new(id, node_type, properties, activation_level)
    end
end

"""
Represents a hyperedge connecting multiple nodes.
"""
struct HypergraphEdge
    id::Int
    edge_type::Symbol  # :perceives, :acts_on, :encapsulated_by, :influences
    nodes::Vector{Int}  # Node IDs connected by this edge
    weight::Float64
    metadata::Dict{Symbol, Any}
    
    function HypergraphEdge(id::Int, edge_type::Symbol, nodes::Vector{Int}; 
                           weight::Float64 = 1.0,
                           metadata::Dict{Symbol, Any} = Dict{Symbol, Any}())
        new(id, edge_type, nodes, weight, metadata)
    end
end

"""
Main hypergraph encoding engine for cognitive pattern representation.
"""
mutable struct HypergraphEncoding
    nodes::Dict{Int, HypergraphNode}
    edges::Dict{Int, HypergraphEdge}
    node_counter::Int
    edge_counter::Int
    type_indices::Dict{Symbol, Vector{Int}}  # node_type -> [node_ids]
    graph_representation::SimpleGraph  # For graph algorithms
    
    function HypergraphEncoding()
        new(
            Dict{Int, HypergraphNode}(),
            Dict{Int, HypergraphEdge}(),
            0,
            0,
            Dict{Symbol, Vector{Int}}(),
            SimpleGraph()
        )
    end
end

"""
    add_node!(encoding, node_type, properties) -> Int

Add a new node to the hypergraph encoding.
"""
function add_node!(encoding::HypergraphEncoding, node_type::Symbol; 
                   properties::Dict{Symbol, Any} = Dict{Symbol, Any}(),
                   activation_level::Float64 = 0.0)
    
    encoding.node_counter += 1
    node_id = encoding.node_counter
    
    node = HypergraphNode(node_id, node_type; 
                         properties=properties, 
                         activation_level=activation_level)
    
    encoding.nodes[node_id] = node
    
    # Update type index
    if !haskey(encoding.type_indices, node_type)
        encoding.type_indices[node_type] = Int[]
    end
    push!(encoding.type_indices[node_type], node_id)
    
    # Add to graph representation
    Graphs.add_vertex!(encoding.graph_representation)
    
    return node_id
end

"""
    add_edge!(encoding, edge_type, node_ids, weight, metadata) -> Int

Add a hyperedge connecting multiple nodes.
"""
function add_edge!(encoding::HypergraphEncoding, edge_type::Symbol, node_ids::Vector{Int}; 
                   weight::Float64 = 1.0,
                   metadata::Dict{Symbol, Any} = Dict{Symbol, Any}())
    
    # Validate that all nodes exist
    for node_id in node_ids
        if !haskey(encoding.nodes, node_id)
            error("Node $node_id does not exist in hypergraph")
        end
    end
    
    encoding.edge_counter += 1
    edge_id = encoding.edge_counter
    
    edge = HypergraphEdge(edge_id, edge_type, node_ids; weight=weight, metadata=metadata)
    encoding.edges[edge_id] = edge
    
    # Add edges to graph representation (for binary connections)
    if length(node_ids) == 2
        # Ensure graph has enough vertices
        while nv(encoding.graph_representation) < max(node_ids...)
            Graphs.add_vertex!(encoding.graph_representation)
        end
        Graphs.add_edge!(encoding.graph_representation, node_ids[1], node_ids[2])
    elseif length(node_ids) > 2
        # For hyperedges, create a star pattern with first node as center
        center = node_ids[1]
        # Ensure graph has enough vertices
        while nv(encoding.graph_representation) < max(node_ids...)
            Graphs.add_vertex!(encoding.graph_representation)
        end
        for i in 2:length(node_ids)
            Graphs.add_edge!(encoding.graph_representation, center, node_ids[i])
        end
    end
    
    return edge_id
end

"""
    encode_agentic_kernel_pattern!(encoding, kernel_id, kernel_type, connections)

Encode an agentic kernel as a hypergraph pattern.
"""
function encode_agentic_kernel_pattern!(encoding::HypergraphEncoding, 
                                       kernel_id::Symbol, 
                                       kernel_type::Symbol,
                                       connections::Vector{Tuple{Symbol, Symbol}})
    
    # Create kernel node
    kernel_properties = Dict{Symbol, Any}(:kernel_id => kernel_id)
    kernel_node_id = add_node!(encoding, :kernel; properties=kernel_properties)
    
    # Create connection nodes and edges based on pattern
    for (connection_type, target) in connections
        if connection_type == :perceives
            # Create percept node
            percept_props = Dict{Symbol, Any}(:percept_type => target)
            percept_node_id = add_node!(encoding, :percept; properties=percept_props)
            
            # Add perceives edge
            add_edge!(encoding, :perceives, [kernel_node_id, percept_node_id])
            
        elseif connection_type == :acts_on
            # Create action node
            action_props = Dict{Symbol, Any}(:action_type => target)
            action_node_id = add_node!(encoding, :action; properties=action_props)
            
            # Add acts_on edge
            add_edge!(encoding, :acts_on, [kernel_node_id, action_node_id])
            
        elseif connection_type == :encapsulated_by
            # Create membrane node
            membrane_props = Dict{Symbol, Any}(:membrane_type => target)
            membrane_node_id = add_node!(encoding, :membrane; properties=membrane_props)
            
            # Add encapsulation edge
            add_edge!(encoding, :encapsulated_by, [kernel_node_id, membrane_node_id])
        end
    end
    
    return kernel_node_id
end

"""
    propagate_activation!(encoding, source_node_id, propagation_factor)

Propagate activation through the hypergraph network.
"""
function propagate_activation!(encoding::HypergraphEncoding, 
                              source_node_id::Int, 
                              propagation_factor::Float64 = 0.8)
    
    if !haskey(encoding.nodes, source_node_id)
        @warn "Source node $source_node_id not found"
        return
    end
    
    # Find all edges connected to source node
    connected_edges = Int[]
    for (edge_id, edge) in encoding.edges
        if source_node_id in edge.nodes
            push!(connected_edges, edge_id)
        end
    end
    
    # Propagate activation to connected nodes
    source_activation = encoding.nodes[source_node_id].activation_level
    
    for edge_id in connected_edges
        edge = encoding.edges[edge_id]
        
        for target_node_id in edge.nodes
            if target_node_id != source_node_id
                # Apply edge weight and propagation factor
                activation_transfer = source_activation * edge.weight * propagation_factor
                
                # Update target node activation
                target_node = encoding.nodes[target_node_id]
                new_activation = target_node.activation_level + activation_transfer
                
                # Create updated node (immutable struct)
                updated_node = HypergraphNode(
                    target_node.id,
                    target_node.node_type;
                    properties=target_node.properties,
                    activation_level=new_activation
                )
                
                encoding.nodes[target_node_id] = updated_node
            end
        end
    end
end

"""
    find_cognitive_patterns(encoding, pattern_type) -> Vector{Vector{Int}}

Find specific cognitive patterns in the hypergraph.
"""
function find_cognitive_patterns(encoding::HypergraphEncoding, pattern_type::Symbol)
    patterns = Vector{Int}[]
    
    if pattern_type == :perception_action_loops
        # Find kernel -> percept -> action chains
        kernel_nodes = get(encoding.type_indices, :kernel, Int[])
        
        for kernel_id in kernel_nodes
            # Find percepts connected to this kernel
            percept_connections = find_connected_nodes(encoding, kernel_id, :percept)
            
            for percept_id in percept_connections
                # Find actions connected to this percept
                action_connections = find_connected_nodes(encoding, percept_id, :action)
                
                for action_id in action_connections
                    push!(patterns, [kernel_id, percept_id, action_id])
                end
            end
        end
        
    elseif pattern_type == :memory_clusters
        # Find memory nodes that are highly interconnected
        memory_nodes = get(encoding.type_indices, :memory, Int[])
        
        for i in 1:length(memory_nodes)
            cluster = [memory_nodes[i]]
            
            for j in (i+1):length(memory_nodes)
                if are_nodes_connected(encoding, memory_nodes[i], memory_nodes[j])
                    push!(cluster, memory_nodes[j])
                end
            end
            
            if length(cluster) > 1
                push!(patterns, cluster)
            end
        end
    end
    
    return patterns
end

"""
    find_connected_nodes(encoding, source_id, target_type) -> Vector{Int}

Find all nodes of target_type connected to source node.
"""
function find_connected_nodes(encoding::HypergraphEncoding, source_id::Int, target_type::Symbol)
    connected = Int[]
    
    for (edge_id, edge) in encoding.edges
        if source_id in edge.nodes
            for node_id in edge.nodes
                if node_id != source_id && haskey(encoding.nodes, node_id)
                    node = encoding.nodes[node_id]
                    if node.node_type == target_type
                        push!(connected, node_id)
                    end
                end
            end
        end
    end
    
    return unique(connected)
end

"""
    are_nodes_connected(encoding, node1_id, node2_id) -> Bool

Check if two nodes are directly connected by any edge.
"""
function are_nodes_connected(encoding::HypergraphEncoding, node1_id::Int, node2_id::Int)
    for (edge_id, edge) in encoding.edges
        if node1_id in edge.nodes && node2_id in edge.nodes
            return true
        end
    end
    return false
end

"""
    generate_mermaid_diagram(encoding) -> String

Generate a Mermaid diagram representation of the hypergraph.
"""
function generate_mermaid_diagram(encoding::HypergraphEncoding)
    lines = ["graph LR"]
    
    # Add nodes
    for (node_id, node) in encoding.nodes
        node_shape = if node.node_type == :kernel
            "(($node_id))"  # Double circle for kernels
        elseif node.node_type == :action
            "[$node_id]"    # Rectangle for actions  
        else
            "($node_id)"    # Circle for others
        end
        
        push!(lines, "    N$node_id$node_shape")
    end
    
    # Add edges
    for (edge_id, edge) in encoding.edges
        if length(edge.nodes) == 2
            node1, node2 = edge.nodes
            edge_label = string(edge.edge_type)
            push!(lines, "    N$node1-->|$edge_label|N$node2")
        end
    end
    
    return join(lines, "\n")
end