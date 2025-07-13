"""
# Distributed Orchestration Mesh

This module implements distributed orchestration with adaptive attention allocation,
load balancing, and membrane-based P-System encapsulation for resilience.
"""

using Distributed

"""
Represents an orchestration node in the distributed mesh.
"""
mutable struct OrchestrationNode{T}
    node_id::Symbol
    kernels::Vector{Symbol}  # Cognitive kernels managed by this node
    load_level::Float64
    attention_capacity::Float64
    membrane_state::Dict{Symbol, Any}  # P-System membrane encapsulation
    network_connections::Vector{Symbol}  # Connected nodes
    
    function OrchestrationNode{T}(node_id::Symbol) where T
        new{T}(
            node_id,
            Symbol[],
            0.0,
            1.0,
            Dict{Symbol, Any}(),
            Symbol[]
        )
    end
end

"""
Main distributed orchestration mesh managing cognitive kernel deployment and routing.
"""
mutable struct OrchestrationMesh{T}
    nodes::Dict{Symbol, OrchestrationNode{T}}
    global_attention_budget::Float64
    load_balancer::Dict{Symbol, Float64}  # kernel -> preferred_load
    redundancy_factor::Int
    ecan_weights::Dict{Symbol, Float64}  # Economic Cognitive Architecture Network weights
    
    function OrchestrationMesh{T}() where T
        new{T}(
            Dict{Symbol, OrchestrationNode{T}}(),
            100.0,  # Total attention budget
            Dict{Symbol, Float64}(),
            2,  # Default redundancy
            Dict{Symbol, Float64}()
        )
    end
end

"""
    add_orchestration_node!(mesh, node_id) -> OrchestrationNode

Add a new orchestration node to the mesh.
"""
function add_orchestration_node!(mesh::OrchestrationMesh{T}, node_id::Symbol) where T
    node = OrchestrationNode{T}(node_id)
    mesh.nodes[node_id] = node
    return node
end

"""
    deploy_kernel_to_node!(mesh, kernel_id, node_id)

Deploy a cognitive kernel to a specific orchestration node.
"""
function deploy_kernel_to_node!(mesh::OrchestrationMesh, kernel_id::Symbol, node_id::Symbol)
    if haskey(mesh.nodes, node_id)
        node = mesh.nodes[node_id]
        push!(node.kernels, kernel_id)
        
        # Update load level based on kernel complexity
        node.load_level += 0.1  # Simple load increment
        
        # Register in load balancer
        mesh.load_balancer[kernel_id] = node.load_level
    else
        @warn "Node $node_id not found in mesh"
    end
end

"""
    allocate_attention!(mesh, activation_levels, usage_patterns)

Allocate attention across the distributed mesh using ECAN principles.
"""
function allocate_attention!(mesh::OrchestrationMesh, 
                           activation_levels::Dict{Symbol, Float64},
                           usage_patterns::Dict{Symbol, Float64})
    
    total_demand = sum(values(activation_levels)) + sum(values(usage_patterns))
    
    if total_demand <= mesh.global_attention_budget
        # Sufficient budget - allocate based on demand
        for (kernel_id, activation) in activation_levels
            weight = activation / total_demand * mesh.global_attention_budget
            mesh.ecan_weights[kernel_id] = weight
        end
    else
        # Insufficient budget - compete for attention
        for (kernel_id, activation) in activation_levels
            # ECAN competition: higher activation and frequency get more attention
            usage = get(usage_patterns, kernel_id, 0.0)
            contextual_relevance = activation * 0.7 + usage * 0.3
            
            # Resource cost consideration
            base_cost = get(mesh.load_balancer, kernel_id, 1.0)
            
            # Final attention score
            attention_score = contextual_relevance / base_cost
            normalized_weight = attention_score / total_demand * mesh.global_attention_budget
            
            mesh.ecan_weights[kernel_id] = normalized_weight
        end
    end
end

"""
    balance_load!(mesh)

Perform load balancing across orchestration nodes.
"""
function balance_load!(mesh::OrchestrationMesh)
    # Calculate average load
    total_load = sum(node.load_level for node in values(mesh.nodes))
    avg_load = total_load / length(mesh.nodes)
    
    # Identify overloaded and underloaded nodes
    overloaded = Symbol[]
    underloaded = Symbol[]
    
    for (node_id, node) in mesh.nodes
        if node.load_level > avg_load * 1.2
            push!(overloaded, node_id)
        elseif node.load_level < avg_load * 0.8
            push!(underloaded, node_id)
        end
    end
    
    # Migrate kernels from overloaded to underloaded nodes
    for over_id in overloaded
        if !isempty(underloaded)
            over_node = mesh.nodes[over_id]
            under_id = pop!(underloaded)
            under_node = mesh.nodes[under_id]
            
            if !isempty(over_node.kernels)
                # Move one kernel
                kernel_to_move = pop!(over_node.kernels)
                push!(under_node.kernels, kernel_to_move)
                
                # Update load levels
                over_node.load_level -= 0.1
                under_node.load_level += 0.1
                
                @info "Migrated kernel $kernel_to_move from $over_id to $under_id"
            end
        end
    end
end

"""
    create_membrane_encapsulation!(node, membrane_type)

Create P-System membrane encapsulation for resilience.
"""
function create_membrane_encapsulation!(node::OrchestrationNode, membrane_type::Symbol)
    membrane = Dict{Symbol, Any}(
        :type => membrane_type,
        :rules => [],
        :objects => node.kernels,
        :permeability => 0.8,
        :created_at => time()
    )
    
    node.membrane_state[membrane_type] = membrane
end

"""
    route_cognitive_message(mesh, from_kernel, to_kernel, message)

Route a cognitive message through the distributed mesh.
"""
function route_cognitive_message(mesh::OrchestrationMesh, 
                                from_kernel::Symbol, 
                                to_kernel::Symbol, 
                                message::Dict{Symbol, Any})
    
    # Find nodes containing the kernels
    from_node_id = find_kernel_node(mesh, from_kernel)
    to_node_id = find_kernel_node(mesh, to_kernel)
    
    if isnothing(from_node_id) || isnothing(to_node_id)
        @warn "Could not find nodes for kernels: $from_kernel -> $to_kernel"
        return false
    end
    
    # Apply attention weighting
    attention_weight = get(mesh.ecan_weights, from_kernel, 0.5) * 
                      get(mesh.ecan_weights, to_kernel, 0.5)
    
    weighted_message = Dict(k => v * attention_weight for (k, v) in message)
    
    @info "Routing message from $from_kernel@$from_node_id to $to_kernel@$to_node_id with weight $attention_weight"
    
    return true
end

"""
    find_kernel_node(mesh, kernel_id) -> Union{Symbol, Nothing}

Find which node contains the specified kernel.
"""
function find_kernel_node(mesh::OrchestrationMesh, kernel_id::Symbol)
    for (node_id, node) in mesh.nodes
        if kernel_id in node.kernels
            return node_id
        end
    end
    return nothing
end