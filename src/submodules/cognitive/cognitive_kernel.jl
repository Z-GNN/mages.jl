"""
# Cognitive Kernel Catalog

This module implements cognitive kernels with prime factorization tensor shapes
and semantic dimension assignment based on agentic function complexity.
"""

"""
Represents a cognitive kernel with tensor shape determined by prime factorization
of its semantic dimensions.
"""
struct CognitiveKernel{T}
    id::Symbol
    function_type::Symbol  # :memory_retrieval, :action_selection, etc.
    semantic_dimensions::Vector{Symbol}
    tensor_shape::Tuple{Vararg{Int}}
    prime_factors::Vector{Int}
    tensor_data::Array{T}
    metadata::Dict{Symbol, Any}
    
    function CognitiveKernel{T}(id::Symbol, function_type::Symbol, semantic_dims::Vector{Symbol}) where T
        shape = assign_tensor_shape(semantic_dims)
        factors = prime_factorize_shape(shape)
        data = zeros(T, shape)
        metadata = Dict{Symbol, Any}()
        
        new{T}(id, function_type, semantic_dims, shape, factors, data, metadata)
    end
end

"""
Registry for managing cognitive kernels with hypergraph metadata.
"""
mutable struct CognitiveKernelCatalog{T}
    kernels::Dict{Symbol, CognitiveKernel{T}}
    hypergraph_links::Vector{Tuple{Symbol, Symbol, Symbol}}  # (from, to, link_type)
    dimension_registry::Dict{Symbol, Int}
    usage_statistics::Dict{Symbol, Int}
    
    function CognitiveKernelCatalog{T}() where T
        new{T}(
            Dict{Symbol, CognitiveKernel{T}}(),
            Tuple{Symbol, Symbol, Symbol}[],
            Dict{Symbol, Int}(),
            Dict{Symbol, Int}()
        )
    end
end

"""
    assign_tensor_shape(semantic_dimensions) -> Tuple{Vararg{Int}}

Assign tensor shape based on semantic complexity and functional depth.
Each semantic dimension gets a size based on its estimated complexity.
"""
function assign_tensor_shape(semantic_dimensions::Vector{Symbol})
    dimension_sizes = Int[]
    
    for dim in semantic_dimensions
        # Assign dimension size based on semantic complexity
        size = if dim == :context
            16  # Context requires more dimensions
        elseif dim == :time
            8   # Temporal dimension
        elseif dim == :salience
            4   # Attention/salience dimension
        elseif dim == :spatial
            12  # Spatial relationships
        elseif dim == :memory
            20  # Memory capacity dimension
        elseif dim == :action
            6   # Action space dimension
        else
            5   # Default dimension size
        end
        push!(dimension_sizes, size)
    end
    
    return Tuple(dimension_sizes)
end

"""
    prime_factorize_shape(shape) -> Vector{Int}

Decompose tensor shape into prime factors for efficient representation.
"""
function prime_factorize_shape(shape::Tuple{Vararg{Int}})
    all_factors = Int[]
    
    for dim_size in shape
        factors = prime_factors(dim_size)
        append!(all_factors, factors)
    end
    
    return all_factors
end

"""
    prime_factors(n) -> Vector{Int}

Get prime factorization of a number.
"""
function prime_factors(n::Int)
    factors = Int[]
    d = 2
    
    while d * d <= n
        while n % d == 0
            push!(factors, d)
            n ÷= d
        end
        d += 1
    end
    
    if n > 1
        push!(factors, n)
    end
    
    return factors
end

"""
    register_kernel!(catalog, kernel_id, function_type, semantic_dims)

Register a new cognitive kernel in the catalog.
"""
function register_kernel!(catalog::CognitiveKernelCatalog{T}, 
                         kernel_id::Symbol, 
                         function_type::Symbol, 
                         semantic_dims::Vector{Symbol}) where T
    
    kernel = CognitiveKernel{T}(kernel_id, function_type, semantic_dims)
    catalog.kernels[kernel_id] = kernel
    catalog.usage_statistics[kernel_id] = 0
    
    # Register semantic dimensions
    for dim in semantic_dims
        if !haskey(catalog.dimension_registry, dim)
            catalog.dimension_registry[dim] = 0
        end
        catalog.dimension_registry[dim] += 1
    end
    
    return kernel
end

"""
    add_hypergraph_link!(catalog, from_kernel, to_kernel, link_type)

Add a hypergraph link between cognitive kernels.
"""
function add_hypergraph_link!(catalog::CognitiveKernelCatalog, 
                             from_kernel::Symbol, 
                             to_kernel::Symbol, 
                             link_type::Symbol)
    link = (from_kernel, to_kernel, link_type)
    push!(catalog.hypergraph_links, link)
end

"""
    get_kernel_interconnections(catalog, kernel_id) -> Vector{Symbol}

Get all kernels connected to the specified kernel in the hypergraph.
"""
function get_kernel_interconnections(catalog::CognitiveKernelCatalog, kernel_id::Symbol)
    connected = Symbol[]
    
    for (from, to, link_type) in catalog.hypergraph_links
        if from == kernel_id
            push!(connected, to)
        elseif to == kernel_id
            push!(connected, from)
        end
    end
    
    return unique(connected)
end

"""
Example: Create a memory retrieval kernel as shown in the issue pseudocode.
"""
function create_memory_retrieval_kernel(catalog::CognitiveKernelCatalog{T}) where T
    semantic_dims = [:context, :time, :salience]
    kernel_id = :memory_retrieval
    
    return register_kernel!(catalog, kernel_id, :memory_retrieval, semantic_dims)
end