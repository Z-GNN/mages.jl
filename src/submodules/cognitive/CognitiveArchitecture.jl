"""
# Cognitive Architecture Module

This module implements a distributed GGML tensor network of agentic cognitive grammar
for Agents.jl, providing cognitive capabilities to agent-based models.

## Core Components

- **AgenticGrammar**: Core grammar adapters for symbolic ↔ subsymbolic mapping
- **TensorNetwork**: GGML-based tensor network layer for distributed cognition
- **OrchestrationMesh**: Distributed orchestration with adaptive attention allocation
- **CognitiveKernel**: Individual cognitive processing units with prime factorization shapes
- **AtomSpaceBridge**: Integration with OpenCog-style hypergraph representations
- **HypergraphEncoding**: Graph-based encoding of cognitive patterns

## Architecture Overview

The cognitive architecture extends Agents.jl with cognitive capabilities while
maintaining full compatibility with existing agent-based modeling functionality.
"""
module CognitiveArchitecture

using ..Agents
using LinearAlgebra
using Graphs
using DataStructures

# Export main types and functions
export AgenticGrammar, TensorNetwork, CognitiveKernel, OrchestrationMesh
export AtomSpaceBridge, HypergraphEncoding, CognitiveKernelCatalog
export TensorBlock, HypergraphNode, HypergraphEdge, Atom, Link, AtomSpace
export ActionPrimitive, PerceptPrimitive, MemoryPrimitive, OrchestrationNode

# Export main functions
export create_cognitive_agent, assign_tensor_shape, prime_factorize_shape
export extract_agentic_primitives, map_to_cognitive_tokens, symbolic_to_subsymbolic
export add_tensor_block!, route_tensor_message, update_attention_weights!, compute_tensor_dynamics
export register_kernel!, add_hypergraph_link!, get_kernel_interconnections, create_memory_retrieval_kernel
export add_orchestration_node!, deploy_kernel_to_node!, allocate_attention!, balance_load!
export route_cognitive_message, find_kernel_node, create_membrane_encapsulation!
export add_atom!, add_link!, sync_kernel_to_atomspace!, sync_tensor_to_atomspace!
export sync_hypergraph_links!, query_atomspace, periodic_sync!
export add_node!, add_edge!, encode_agentic_kernel_pattern!, propagate_activation!
export find_cognitive_patterns, find_connected_nodes, are_nodes_connected, generate_mermaid_diagram

# Include submodule components
include("agentic_grammar.jl")
include("tensor_network.jl") 
include("cognitive_kernel.jl")
include("orchestration_mesh.jl")
include("atomspace_bridge.jl")
include("hypergraph_encoding.jl")

end # module CognitiveArchitecture