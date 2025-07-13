"""
Example: Cognitive Navigation Agent

This example demonstrates the distributed GGML tensor network of agentic cognitive grammar
in action with a simple navigation agent that uses cognitive primitives for spatial reasoning.
"""

using Agents
using Agents.CognitiveArchitecture
using Statistics

# Define a cognitive navigation agent
@agent struct CognitiveNavigationAgent(GridAgent{2})
    energy::Float64
    target_position::Tuple{Int, Int}
    memory_buffer::Vector{Tuple{Int, Int}}  # Stores recently visited positions
    cognitive_state::Dict{Symbol, Float64}  # Internal cognitive variables
end

# Agent step function with cognitive processing
function cognitive_navigation_step!(agent, model)
    # Extract agentic primitives from current state
    primitives = extract_agentic_primitives(agent, model)
    
    # Create cognitive grammar and process primitives
    grammar = AgenticGrammar{Float64}()
    tokens = map_to_cognitive_tokens(grammar, primitives)
    weights = symbolic_to_subsymbolic(grammar, tokens)
    
    # Update cognitive state based on tensor weights
    agent.cognitive_state[:perception_strength] = mean(weights)
    agent.cognitive_state[:memory_activation] = length(agent.memory_buffer) / 10.0
    
    # Cognitive decision making for movement
    current_pos = agent.pos
    target_pos = agent.target_position
    
    # Calculate direction towards target using cognitive weighting
    dx = target_pos[1] - current_pos[1]
    dy = target_pos[2] - current_pos[2]
    
    # Apply cognitive bias based on memory and perception
    memory_bias = agent.cognitive_state[:memory_activation] * 0.1
    perception_bias = agent.cognitive_state[:perception_strength] * 0.2
    
    # Determine next position with cognitive influence
    next_x = current_pos[1] + sign(dx) * (1 + memory_bias)
    next_y = current_pos[2] + sign(dy) * (1 + perception_bias)
    
    # Ensure position is within bounds
    next_x = clamp(round(Int, next_x), 1, size(abmspace(model))[1])
    next_y = clamp(round(Int, next_y), 1, size(abmspace(model))[2])
    
    # Move agent
    move_agent!(agent, (next_x, next_y), model)
    
    # Update memory buffer
    push!(agent.memory_buffer, current_pos)
    if length(agent.memory_buffer) > 5  # Keep only recent 5 positions
        popfirst!(agent.memory_buffer)
    end
    
    # Update energy based on cognitive processing
    agent.energy -= 0.1 * (1 + agent.cognitive_state[:perception_strength])
    
    # Update target if reached
    if agent.pos == agent.target_position
        # Set new random target
        space_size = size(abmspace(model))
        agent.target_position = (rand(1:space_size[1]), rand(1:space_size[2]))
        agent.energy += 5.0  # Reward for reaching target
    end
end

# Create and run the cognitive navigation example
function run_cognitive_navigation_example()
    println("🧠 Cognitive Navigation Agent Example")
    println("=" ^ 50)
    
    # Create model with cognitive navigation agents
    model = StandardABM(
        CognitiveNavigationAgent, 
        GridSpace((10, 10)),
        agent_step! = cognitive_navigation_step!
    )
    
    # Add cognitive agents
    for i in 1:3
        agent = CognitiveNavigationAgent(
            i, 
            (rand(1:10), rand(1:10)),  # Random starting position
            100.0,  # Initial energy
            (rand(1:10), rand(1:10)),  # Random target
            Tuple{Int, Int}[],  # Empty memory buffer
            Dict{Symbol, Float64}()  # Empty cognitive state
        )
        add_agent!(agent, model)
    end
    
    # Create comprehensive cognitive architecture for the model
    println("\n1. Creating Cognitive Architecture Components...")
    
    # Create tensor network for distributed cognitive processing
    tensor_network = TensorNetwork{Float64}()
    
    # Create cognitive kernel catalog
    kernel_catalog = CognitiveKernelCatalog{Float64}()
    
    # Create orchestration mesh for distributed processing
    orch_mesh = OrchestrationMesh{Float64}()
    add_orchestration_node!(orch_mesh, :main_cognitive_node)
    
    # Create AtomSpace bridge for hypergraph representation
    atomspace_bridge = AtomSpaceBridge{Float64}()
    
    # Create hypergraph encoding for cognitive patterns
    hypergraph = HypergraphEncoding()
    
    # Set up cognitive kernels for different functions
    cognitive_functions = [
        (:spatial_perception, :perception, [:spatial, :context]),
        (:memory_retrieval, :memory, [:context, :time, :salience]),
        (:action_selection, :action, [:spatial, :time]),
        (:target_seeking, :navigation, [:spatial, :salience]),
        (:energy_management, :homeostasis, [:time, :salience])
    ]
    
    println("\n2. Registering Cognitive Kernels...")
    for (kernel_id, function_type, semantic_dims) in cognitive_functions
        # Register in catalog
        kernel = register_kernel!(kernel_catalog, kernel_id, function_type, semantic_dims)
        println("   ✓ Registered $kernel_id with shape $(kernel.tensor_shape)")
        
        # Add to tensor network
        add_tensor_block!(tensor_network, kernel_id, kernel.tensor_shape)
        
        # Deploy to orchestration mesh
        deploy_kernel_to_node!(orch_mesh, kernel_id, :main_cognitive_node)
        
        # Sync to AtomSpace
        metadata = Dict{Symbol, Any}(:function_type => string(function_type))
        sync_kernel_to_atomspace!(atomspace_bridge, kernel_id, function_type, metadata)
        
        # Encode in hypergraph
        connections = [(:processes, function_type), (:influences, :behavior)]
        encode_agentic_kernel_pattern!(hypergraph, kernel_id, function_type, connections)
    end
    
    # Add cognitive interdependencies
    println("\n3. Creating Cognitive Interdependencies...")
    interdependencies = [
        (:spatial_perception, :memory_retrieval, :feeds_into),
        (:memory_retrieval, :action_selection, :influences),
        (:action_selection, :target_seeking, :guides),
        (:target_seeking, :energy_management, :affects),
        (:energy_management, :spatial_perception, :modulates)
    ]
    
    for (from_kernel, to_kernel, link_type) in interdependencies
        add_hypergraph_link!(kernel_catalog, from_kernel, to_kernel, link_type)
        println("   ✓ $from_kernel --$link_type--> $to_kernel")
    end
    
    # Sync hypergraph links to AtomSpace
    sync_hypergraph_links!(atomspace_bridge, kernel_catalog.hypergraph_links)
    
    println("\n4. Running Cognitive Simulation...")
    
    # Run simulation steps
    for step in 1:10
        println("\n--- Step $step ---")
        
        # Step all agents
        step!(model)
        
        # Compute cognitive dynamics
        activations = compute_tensor_dynamics(tensor_network)
        println("Tensor activations: $(round.(activations, digits=3))")
        
        # Update attention allocation using ECAN
        activation_dict = Dict(kernel_id => activations[i] 
                             for (i, kernel_id) in enumerate(keys(tensor_network.kernel_registry)))
        usage_patterns = Dict(kernel_id => rand() for kernel_id in keys(tensor_network.kernel_registry))
        
        allocate_attention!(orch_mesh, activation_dict, usage_patterns)
        
        # Load balancing if needed
        if step % 3 == 0
            balance_load!(orch_mesh)
        end
        
        # Display agent states
        for agent in allagents(model)
            cognitive_strength = get(agent.cognitive_state, :perception_strength, 0.0)
            memory_level = get(agent.cognitive_state, :memory_activation, 0.0)
            println("   Agent $(agent.id): pos=$(agent.pos), target=$(agent.target_position), " *
                   "energy=$(round(agent.energy, digits=1)), " *
                   "cognitive=$(round(cognitive_strength, digits=3)), " *
                   "memory=$(round(memory_level, digits=3))")
        end
        
        # Periodic AtomSpace synchronization
        if periodic_sync!(atomspace_bridge)
            println("   ✓ AtomSpace synchronized")
        end
    end
    
    # Display final cognitive architecture statistics
    println("\n5. Final Cognitive Architecture State:")
    println("   📊 Tensor Network: $(length(tensor_network.blocks)) active blocks")
    println("   🧠 Kernel Catalog: $(length(kernel_catalog.kernels)) registered kernels")
    println("   🌐 Orchestration Mesh: $(length(orch_mesh.nodes)) nodes, " *
           "$(sum(length(node.kernels) for node in values(orch_mesh.nodes))) deployed kernels")
    println("   🔗 AtomSpace: $(length(atomspace_bridge.atomspace.atoms)) atoms, " *
           "$(length(atomspace_bridge.atomspace.links)) links")
    println("   📈 Hypergraph: $(length(hypergraph.nodes)) nodes, $(length(hypergraph.edges)) edges")
    
    # Query AtomSpace for cognitive patterns
    cognitive_kernels = query_atomspace(atomspace_bridge, :by_type, "CognitiveKernel")
    println("   🔍 AtomSpace cognitive kernels found: $(length(cognitive_kernels))")
    
    # Find cognitive patterns in hypergraph
    perception_loops = find_cognitive_patterns(hypergraph, :perception_action_loops)
    println("   🔄 Perception-action loops detected: $(length(perception_loops))")
    
    # Generate Mermaid diagram of the cognitive architecture
    println("\n6. Cognitive Architecture Diagram:")
    mermaid_diagram = generate_mermaid_diagram(hypergraph)
    println(mermaid_diagram)
    
    println("\n🎯 Cognitive navigation example completed successfully!")
    println("The agents demonstrated distributed GGML tensor network of agentic cognitive grammar")
    println("with prime factorization tensor shapes, ECAN attention allocation, and hypergraph encoding.")
    
    return model, tensor_network, kernel_catalog, orch_mesh, atomspace_bridge, hypergraph
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    # Run the cognitive navigation example
    model, tensor_network, kernel_catalog, orch_mesh, atomspace_bridge, hypergraph = run_cognitive_navigation_example()
    
    println("\n" * "=" ^ 70)
    println("Cognitive architecture components are available in the returned tuple:")
    println("- model: Agent-based model with cognitive navigation agents")  
    println("- tensor_network: GGML-compatible tensor network")
    println("- kernel_catalog: Cognitive kernel registry with prime factorization")
    println("- orch_mesh: Distributed orchestration mesh with ECAN attention")
    println("- atomspace_bridge: OpenCog AtomSpace integration")
    println("- hypergraph: Hypergraph encoding of cognitive patterns")
    println("=" ^ 70)
end