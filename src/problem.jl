"""
    Problem(num_layers::Int, local_fields::Vector{Real}, couplings::Matrix{Real}, driver)

A container holding the basic properties of the QAOA circuit.

$(TYPEDFIELDS)
"""
@with_kw struct Problem
    "The number of qubits of the QAOA circuit."
    num_qubits::Int

    "The number of layers ``p`` of the QAOA circuit."
    num_layers::Int

    "The local (magnetic) fields of the Ising problem Hamiltonian."
    local_fields::Vector{Real}

    "The coupling matrix of the Ising problem Hamiltonian."
    couplings::Matrix{Real}

    "The edges of the graph specified by the coupling matrix."
    edges

    "The driver of the QAOA circuit. By default the Pauli matrix `X`. May also be set to, 
    e.g., `[[X, X], [Y, Y]]` to obtain the drivers``\\hat X_i \\hat X_j + \\hat Y_i \\hat Y_j`` acting on every pair of connected qubits."
    driver

    # The default constructor
    Problem(num_layers, local_fields, couplings) = new(size(local_fields)[1], 
                                                       num_layers, 
                                                       local_fields, 
                                                       couplings, 
                                                       findall(x -> !iszero(x), couplings), 
                                                       X)

    # Constructor with customized driver
    Problem(num_layers, local_fields, couplings, driver) = new(size(local_fields)[1], 
                                                               num_layers, 
                                                               local_fields, 
                                                               couplings, 
                                                               findall(x -> !iszero(x), couplings), 
                                                               driver)

    # Default constructor explicitly breaking the Z2 symmetry when the local fields are zero
    Problem(num_layers, couplings) = new(size(couplings)[1] - 1, 
                                         num_layers, 
                                         couplings[1:end-1, end], 
                                         couplings[1:end-1, 1:end-1], 
                                         findall(x -> !iszero(x), couplings[1:end-1, 1:end-1]), 
                                         X)
end







