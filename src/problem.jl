@with_kw struct Problem
    num_qubits::Int
    num_layers::Int
    local_fields::Vector{Real}
    couplings::Matrix{Real}
    driver
    Problem(num_layers, local_fields, couplings) = new(size(local_fields)[1], num_layers, local_fields, couplings, X)
    Problem(num_layers, local_fields, couplings, driver) = new(size(local_fields)[1], num_layers, local_fields, couplings, driver)
end







