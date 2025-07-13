"""
# Agentic Grammar Adapters

This module implements grammar extraction and symbolic ↔ subsymbolic mapping
for cognitive agent behaviors, bridging traditional agent constructs to 
cognitive grammar tokens.
"""

# Core agentic grammar primitives
abstract type AgenticPrimitive end

struct ActionPrimitive <: AgenticPrimitive
    name::Symbol
    parameters::Dict{Symbol, Any}
    cognitive_weight::Float64
end

struct PerceptPrimitive <: AgenticPrimitive
    sensor_type::Symbol
    data_type::Type
    attention_level::Float64
end

struct MemoryPrimitive <: AgenticPrimitive
    memory_type::Symbol  # :working, :episodic, :semantic
    capacity::Int
    decay_rate::Float64
end

"""
Main agentic grammar adapter that extracts cognitive primitives from agent behaviors
and maps them to tensor-compatible representations.
"""
struct AgenticGrammar{T}
    primitives::Vector{AgenticPrimitive}
    symbolic_mapping::Dict{Symbol, Vector{T}}
    subsymbolic_weights::Vector{Float64}
    
    function AgenticGrammar{T}() where T
        new{T}(
            AgenticPrimitive[],
            Dict{Symbol, Vector{T}}(),
            Float64[]
        )
    end
end

"""
    extract_agentic_primitives(agent, model) -> Vector{AgenticPrimitive}

Extract agentic primitives (actions, percepts, memory operations) from an agent
in the context of its model environment.
"""
function extract_agentic_primitives(agent, model)
    primitives = AgenticPrimitive[]
    
    # Extract action primitives from agent methods
    for method in fieldnames(typeof(agent))
        if string(method) |> contains("step") || string(method) |> contains("act")
            push!(primitives, ActionPrimitive(method, Dict(), 1.0))
        end
    end
    
    # Extract percept primitives from model space interaction
    if hasfield(typeof(model), :space) && !isnothing(getfield(model, :space))
        push!(primitives, PerceptPrimitive(:spatial, typeof(getfield(model, :space)), 0.8))
    end
    
    # Add basic memory primitive
    push!(primitives, MemoryPrimitive(:working, 100, 0.1))
    
    return primitives
end

"""
    map_to_cognitive_tokens(grammar::AgenticGrammar, primitives) -> Vector{Symbol}

Convert agentic primitives to cognitive grammar tokens for tensor processing.
"""
function map_to_cognitive_tokens(grammar::AgenticGrammar{T}, primitives) where T
    tokens = Symbol[]
    
    for primitive in primitives
        if primitive isa ActionPrimitive
            push!(tokens, Symbol("action_$(primitive.name)"))
        elseif primitive isa PerceptPrimitive
            push!(tokens, Symbol("percept_$(primitive.sensor_type)"))
        elseif primitive isa MemoryPrimitive
            push!(tokens, Symbol("memory_$(primitive.memory_type)"))
        end
    end
    
    return tokens
end

"""
    symbolic_to_subsymbolic(grammar::AgenticGrammar, tokens) -> Vector{Float64}

Bridge symbolic cognitive tokens to subsymbolic tensor representations.
"""
function symbolic_to_subsymbolic(grammar::AgenticGrammar{T}, tokens) where T
    # Create subsymbolic representation as weighted vector
    weights = Float64[]
    
    for token in tokens
        # Simple mapping - could be enhanced with learned embeddings
        weight = hash(token) % 1000 / 1000.0  # Normalize to [0,1]
        push!(weights, weight)
    end
    
    return weights
end