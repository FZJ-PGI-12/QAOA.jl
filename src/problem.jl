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

    "The driver of the QAOA circuit. By default the Pauli matrix `X`. May also be set to, e.g., `[[X, X], [Y, Y]]` to obtain the driver ``\\hat X_i \\hat X_j + \\hat Y_i \\hat Y_j``."
    driver

    Problem(num_layers, local_fields, couplings) = new(size(local_fields)[1], num_layers, local_fields, couplings, X)
    Problem(num_layers, local_fields, couplings, driver) = new(size(local_fields)[1], num_layers, local_fields, couplings, driver)
end







