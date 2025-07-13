"""
# OpenCog AtomSpace Bridge

This module implements bidirectional synchronization with OpenCog-style AtomSpace
for hypergraph representation of cognitive kernel interconnections.
"""

"""
Represents an atom in the AtomSpace with type and truth value.
"""
struct Atom
    id::UInt64
    atom_type::Symbol
    name::String
    truth_value::Tuple{Float64, Float64}  # (strength, confidence)
    outgoing::Vector{UInt64}  # Links to other atoms
    
    function Atom(atom_type::Symbol, name::String; 
                  truth_value::Tuple{Float64, Float64} = (1.0, 1.0),
                  outgoing::Vector{UInt64} = UInt64[])
        id = hash((atom_type, name, time())) % typemax(UInt64)
        new(id, atom_type, name, truth_value, outgoing)
    end
end

"""
Represents a link between atoms in the hypergraph.
"""
struct Link
    id::UInt64
    link_type::Symbol
    outgoing::Vector{UInt64}  # Atoms this link connects
    truth_value::Tuple{Float64, Float64}
    
    function Link(link_type::Symbol, outgoing::Vector{UInt64}; 
                  truth_value::Tuple{Float64, Float64} = (1.0, 1.0))
        id = hash((link_type, outgoing, time())) % typemax(UInt64)
        new(id, link_type, outgoing, truth_value)
    end
end

"""
AtomSpace implementation for cognitive kernel hypergraph representation.
"""
mutable struct AtomSpace
    atoms::Dict{UInt64, Atom}
    links::Dict{UInt64, Link}
    type_index::Dict{Symbol, Vector{UInt64}}  # atom_type -> [atom_ids]
    name_index::Dict{String, UInt64}  # name -> atom_id
    
    function AtomSpace()
        new(
            Dict{UInt64, Atom}(),
            Dict{UInt64, Link}(),
            Dict{Symbol, Vector{UInt64}}(),
            Dict{String, UInt64}()
        )
    end
end

"""
Bridge between cognitive kernels and AtomSpace representation.
"""
mutable struct AtomSpaceBridge{T}
    atomspace::AtomSpace
    kernel_to_atom::Dict{Symbol, UInt64}  # kernel_id -> atom_id
    tensor_to_atom::Dict{TensorBlock{T}, UInt64}  # tensor_block -> atom_id
    sync_frequency::Float64
    last_sync::Float64
    
    function AtomSpaceBridge{T}() where T
        new{T}(
            AtomSpace(),
            Dict{Symbol, UInt64}(),
            Dict{TensorBlock{T}, UInt64}(),
            1.0,  # Sync every second
            time()
        )
    end
end

"""
    add_atom!(atomspace, atom_type, name, truth_value) -> UInt64

Add a new atom to the AtomSpace.
"""
function add_atom!(atomspace::AtomSpace, atom_type::Symbol, name::String; 
                   truth_value::Tuple{Float64, Float64} = (1.0, 1.0))
    
    atom = Atom(atom_type, name; truth_value=truth_value)
    
    atomspace.atoms[atom.id] = atom
    atomspace.name_index[name] = atom.id
    
    # Update type index
    if !haskey(atomspace.type_index, atom_type)
        atomspace.type_index[atom_type] = UInt64[]
    end
    push!(atomspace.type_index[atom_type], atom.id)
    
    return atom.id
end

"""
    add_link!(atomspace, link_type, outgoing_atoms, truth_value) -> UInt64

Add a new link between atoms in the AtomSpace.
"""
function add_link!(atomspace::AtomSpace, link_type::Symbol, outgoing_atoms::Vector{UInt64}; 
                   truth_value::Tuple{Float64, Float64} = (1.0, 1.0))
    
    link = Link(link_type, outgoing_atoms; truth_value=truth_value)
    atomspace.links[link.id] = link
    
    return link.id
end

"""
    sync_kernel_to_atomspace!(bridge, kernel_id, kernel_type, metadata)

Synchronize a cognitive kernel to AtomSpace representation.
"""
function sync_kernel_to_atomspace!(bridge::AtomSpaceBridge, 
                                  kernel_id::Symbol, 
                                  kernel_type::Symbol, 
                                  metadata::Dict{Symbol, Any})
    
    # Create atom for the kernel if it doesn't exist
    if !haskey(bridge.kernel_to_atom, kernel_id)
        atom_id = add_atom!(bridge.atomspace, :CognitiveKernel, string(kernel_id))
        bridge.kernel_to_atom[kernel_id] = atom_id
        
        # Add type information as a link
        type_atom_id = add_atom!(bridge.atomspace, :ConceptNode, string(kernel_type))
        add_link!(bridge.atomspace, :InheritanceLink, [atom_id, type_atom_id])
    end
    
    # Sync metadata as additional atoms and links
    kernel_atom_id = bridge.kernel_to_atom[kernel_id]
    
    for (key, value) in metadata
        prop_atom_id = add_atom!(bridge.atomspace, :ConceptNode, "$(key)_$(value)")
        add_link!(bridge.atomspace, :EvaluationLink, [kernel_atom_id, prop_atom_id])
    end
end

"""
    sync_tensor_to_atomspace!(bridge, tensor_block, kernel_id)

Synchronize tensor block state to AtomSpace.
"""
function sync_tensor_to_atomspace!(bridge::AtomSpaceBridge{T}, 
                                  tensor_block::TensorBlock{T}, 
                                  kernel_id::Symbol) where T
    
    # Create atom for tensor if it doesn't exist
    if !haskey(bridge.tensor_to_atom, tensor_block)
        tensor_name = "tensor_$(kernel_id)_$(hash(tensor_block.shape))"
        atom_id = add_atom!(bridge.atomspace, :TensorNode, tensor_name)
        bridge.tensor_to_atom[tensor_block] = atom_id
        
        # Link tensor to its kernel
        if haskey(bridge.kernel_to_atom, kernel_id)
            kernel_atom_id = bridge.kernel_to_atom[kernel_id]
            add_link!(bridge.atomspace, :EvaluationLink, [kernel_atom_id, atom_id])
        end
    end
    
    # Update tensor state as truth value (simplified representation)
    tensor_atom_id = bridge.tensor_to_atom[tensor_block]
    tensor_norm = norm(tensor_block.data)
    
    # Update truth value based on tensor activation
    strength = min(tensor_norm / 10.0, 1.0)  # Normalize to [0,1]
    confidence = 0.9  # High confidence in tensor state
    
    if haskey(bridge.atomspace.atoms, tensor_atom_id)
        atom = bridge.atomspace.atoms[tensor_atom_id]
        updated_atom = Atom(atom.atom_type, atom.name; 
                           truth_value=(strength, confidence), 
                           outgoing=atom.outgoing)
        bridge.atomspace.atoms[tensor_atom_id] = updated_atom
    end
end

"""
    sync_hypergraph_links!(bridge, hypergraph_links)

Synchronize hypergraph links from cognitive kernel catalog to AtomSpace.
"""
function sync_hypergraph_links!(bridge::AtomSpaceBridge, 
                               hypergraph_links::Vector{Tuple{Symbol, Symbol, Symbol}})
    
    for (from_kernel, to_kernel, link_type) in hypergraph_links
        # Ensure both kernels have atoms
        from_atom_id = get(bridge.kernel_to_atom, from_kernel, nothing)
        to_atom_id = get(bridge.kernel_to_atom, to_kernel, nothing)
        
        if !isnothing(from_atom_id) && !isnothing(to_atom_id)
            # Create link between kernel atoms
            atomspace_link_type = Symbol("$(link_type)Link")
            add_link!(bridge.atomspace, atomspace_link_type, [from_atom_id, to_atom_id])
        end
    end
end

"""
    query_atomspace(bridge, query_type, pattern) -> Vector{UInt64}

Query the AtomSpace for atoms matching a pattern.
"""
function query_atomspace(bridge::AtomSpaceBridge, query_type::Symbol, pattern::String)
    results = UInt64[]
    
    if query_type == :by_type
        atom_type = Symbol(pattern)
        results = get(bridge.atomspace.type_index, atom_type, UInt64[])
    elseif query_type == :by_name
        atom_id = get(bridge.atomspace.name_index, pattern, nothing)
        if !isnothing(atom_id)
            push!(results, atom_id)
        end
    end
    
    return results
end

"""
    periodic_sync!(bridge)

Perform periodic synchronization if enough time has passed.
"""
function periodic_sync!(bridge::AtomSpaceBridge)
    current_time = time()
    if current_time - bridge.last_sync >= bridge.sync_frequency
        @info "Performing periodic AtomSpace synchronization"
        bridge.last_sync = current_time
        return true
    end
    return false
end