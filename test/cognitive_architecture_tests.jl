"""
Test suite for the Cognitive Architecture module.

Tests the basic functionality of all cognitive components including
agentic grammar, tensor networks, cognitive kernels, orchestration mesh,
AtomSpace bridge, and hypergraph encoding.
"""

using Test
using Agents
using Agents.CognitiveArchitecture
using LinearAlgebra

# Basic test agent type
@agent struct CognitiveTestAgent(GridAgent{2})
    energy::Float64
end

@testset "Cognitive Architecture Tests" begin
    
    @testset "Agentic Grammar" begin
        # Create test model and agent
        model = StandardABM(CognitiveTestAgent, GridSpace((5, 5)))
        agent = CognitiveTestAgent(1, (1, 1), 100.0)
        add_agent!(agent, model)
        
        # Test primitive extraction
        primitives = extract_agentic_primitives(agent, model)
        @test length(primitives) >= 2  # Should have at least spatial percept and memory
        
        # Test grammar creation and token mapping
        grammar = AgenticGrammar{Float64}()
        tokens = map_to_cognitive_tokens(grammar, primitives)
        @test length(tokens) == length(primitives)
        
        # Test symbolic to subsymbolic mapping
        weights = symbolic_to_subsymbolic(grammar, tokens)
        @test length(weights) == length(tokens)
        @test all(w -> 0 <= w <= 1, weights)
    end
    
    @testset "Tensor Network" begin
        # Create tensor network
        network = TensorNetwork{Float64}()
        @test length(network.blocks) == 0
        
        # Add tensor blocks
        block_idx = add_tensor_block!(network, :test_kernel, (4, 4))
        @test block_idx == 1
        @test length(network.blocks) == 1
        @test haskey(network.kernel_registry, :test_kernel)
        
        # Add second block
        add_tensor_block!(network, :test_kernel2, (3, 3))
        @test length(network.blocks) == 2
        @test size(network.routing_matrix) == (2, 2)
        
        # Test message routing
        message = rand(Float64, 4, 4)
        route_tensor_message(network, :test_kernel, :test_kernel2, message)
        
        # Test dynamics computation
        activations = compute_tensor_dynamics(network)
        @test length(activations) == 2
    end
    
    @testset "Cognitive Kernel Catalog" begin
        # Create catalog
        catalog = CognitiveKernelCatalog{Float64}()
        @test length(catalog.kernels) == 0
        
        # Test tensor shape assignment
        semantic_dims = [:context, :time, :salience]
        shape = assign_tensor_shape(semantic_dims)
        @test shape == (16, 8, 4)
        
        # Test prime factorization
        factors = prime_factorize_shape(shape)
        @test length(factors) > 0
        @test all(f -> f > 1, factors)  # All should be prime numbers
        
        # Register kernel
        kernel = register_kernel!(catalog, :test_kernel, :memory, semantic_dims)
        @test kernel.id == :test_kernel
        @test kernel.semantic_dimensions == semantic_dims
        @test kernel.tensor_shape == shape
        @test haskey(catalog.kernels, :test_kernel)
        
        # Test hypergraph links
        register_kernel!(catalog, :test_kernel2, :action, [:spatial])
        add_hypergraph_link!(catalog, :test_kernel, :test_kernel2, :influences)
        @test length(catalog.hypergraph_links) == 1
        
        # Test interconnections
        connections = get_kernel_interconnections(catalog, :test_kernel)
        @test :test_kernel2 in connections
        
        # Test memory retrieval kernel example
        memory_kernel = create_memory_retrieval_kernel(catalog)
        @test memory_kernel.function_type == :memory_retrieval
        @test memory_kernel.semantic_dimensions == [:context, :time, :salience]
    end
    
    @testset "Orchestration Mesh" begin
        # Create mesh
        mesh = OrchestrationMesh{Float64}()
        @test length(mesh.nodes) == 0
        
        # Add orchestration node
        node = add_orchestration_node!(mesh, :node1)
        @test node.node_id == :node1
        @test haskey(mesh.nodes, :node1)
        
        # Deploy kernel to node
        deploy_kernel_to_node!(mesh, :test_kernel, :node1)
        @test :test_kernel in mesh.nodes[:node1].kernels
        @test mesh.nodes[:node1].load_level > 0
        
        # Test attention allocation
        activations = Dict(:test_kernel => 0.8)
        usage_patterns = Dict(:test_kernel => 0.9)
        allocate_attention!(mesh, activations, usage_patterns)
        @test haskey(mesh.ecan_weights, :test_kernel)
        @test mesh.ecan_weights[:test_kernel] > 0
        
        # Test load balancing
        add_orchestration_node!(mesh, :node2)
        balance_load!(mesh)
        
        # Test message routing
        success = route_cognitive_message(mesh, :test_kernel, :test_kernel, Dict{Symbol, Any}(:data => 1.0))
        @test success == true
    end
    
    @testset "AtomSpace Bridge" begin
        # Create bridge
        bridge = AtomSpaceBridge{Float64}()
        @test length(bridge.atomspace.atoms) == 0
        
        # Add atoms
        atom_id = add_atom!(bridge.atomspace, :TestNode, "test_atom")
        @test haskey(bridge.atomspace.atoms, atom_id)
        @test bridge.atomspace.name_index["test_atom"] == atom_id
        
        # Add links
        atom2_id = add_atom!(bridge.atomspace, :TestNode, "test_atom2")
        link_id = add_link!(bridge.atomspace, :TestLink, [atom_id, atom2_id])
        @test haskey(bridge.atomspace.links, link_id)
        
        # Test kernel synchronization
        metadata = Dict{Symbol, Any}(:test_prop => "test_value")
        sync_kernel_to_atomspace!(bridge, :test_kernel, :memory, metadata)
        @test haskey(bridge.kernel_to_atom, :test_kernel)
        
        # Test tensor synchronization
        tensor_block = TensorBlock{Float64}((3, 3))
        tensor_block.data .= rand(3, 3)
        sync_tensor_to_atomspace!(bridge, tensor_block, :test_kernel)
        
        # Test queries
        concept_nodes = query_atomspace(bridge, :by_type, "ConceptNode")
        @test length(concept_nodes) > 0
        
        test_atom = query_atomspace(bridge, :by_name, "test_atom")
        @test length(test_atom) == 1
    end
    
    @testset "Hypergraph Encoding" begin
        # Create encoding
        encoding = HypergraphEncoding()
        @test length(encoding.nodes) == 0
        
        # Add nodes
        node1 = add_node!(encoding, :kernel; properties=Dict{Symbol, Any}(:name => "test"))
        node2 = add_node!(encoding, :percept)
        @test haskey(encoding.nodes, node1)
        @test haskey(encoding.nodes, node2)
        @test encoding.nodes[node1].node_type == :kernel
        
        # Add edges
        edge_id = CognitiveArchitecture.add_edge!(encoding, :perceives, [node1, node2])
        @test haskey(encoding.edges, edge_id)
        @test encoding.edges[edge_id].nodes == [node1, node2]
        
        # Test agentic kernel pattern encoding
        connections = [(:perceives, :spatial), (:acts_on, :movement)]
        kernel_node = encode_agentic_kernel_pattern!(encoding, :nav_kernel, :navigation, connections)
        @test haskey(encoding.nodes, kernel_node)
        
        # Test activation propagation
        encoding.nodes[kernel_node] = HypergraphNode(
            kernel_node, :kernel; 
            properties=encoding.nodes[kernel_node].properties,
            activation_level=1.0
        )
        propagate_activation!(encoding, kernel_node, 0.8)
        
        # Test pattern finding
        patterns = find_cognitive_patterns(encoding, :perception_action_loops)
        @test isa(patterns, Vector{Vector{Int}})
        
        # Test connectivity
        connected = find_connected_nodes(encoding, kernel_node, :percept)
        @test isa(connected, Vector{Int})
        
        # Test Mermaid generation
        mermaid_diagram = generate_mermaid_diagram(encoding)
        @test occursin("graph LR", mermaid_diagram)
        @test occursin("N$kernel_node", mermaid_diagram)
    end
    
    @testset "Integration Test" begin
        # Full integration test combining all components
        model = StandardABM(CognitiveTestAgent, GridSpace((5, 5)))
        agent = CognitiveTestAgent(1, (1, 1), 100.0)
        add_agent!(agent, model)
        
        # Extract primitives and create grammar
        primitives = extract_agentic_primitives(agent, model)
        grammar = AgenticGrammar{Float64}()
        tokens = map_to_cognitive_tokens(grammar, primitives)
        
        # Create tensor network and add blocks
        network = TensorNetwork{Float64}()
        for token in tokens
            shape = assign_tensor_shape([Symbol("semantic_$(hash(token) % 3)")])
            add_tensor_block!(network, token, shape)
        end
        
        # Create catalog and register kernels
        catalog = CognitiveKernelCatalog{Float64}()
        for token in tokens
            register_kernel!(catalog, token, :general, [Symbol("dim_$(hash(token) % 3)")])
        end
        
        # Create orchestration mesh
        mesh = OrchestrationMesh{Float64}()
        add_orchestration_node!(mesh, :main_node)
        for token in tokens
            deploy_kernel_to_node!(mesh, token, :main_node)
        end
        
        # Create AtomSpace bridge and sync
        bridge = AtomSpaceBridge{Float64}()
        for token in tokens
            sync_kernel_to_atomspace!(bridge, token, :general, Dict{Symbol, Any}(:token => string(token)))
        end
        
        # Create hypergraph encoding
        encoding = HypergraphEncoding()
        for token in tokens
            connections = [(:processes, :data)]
            encode_agentic_kernel_pattern!(encoding, token, :general, connections)
        end
        
        # Verify integration
        @test length(network.blocks) == length(tokens)
        @test length(catalog.kernels) == length(tokens)
        @test length(mesh.nodes[:main_node].kernels) == length(tokens)
        @test length(bridge.kernel_to_atom) == length(tokens)
        @test length(encoding.nodes) >= length(tokens)
        
        println("✓ Full cognitive architecture integration test passed")
    end
end

println("All cognitive architecture tests completed successfully!")