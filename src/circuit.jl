function local_field_gates(num_qubits::Int, gate::Function=Rz)
    return [put(i => gate(0.))(num_qubits) for i in 1:num_qubits] |> chain
end

function local_field_gates(num_qubits::Int, operator=Z)
    return [put(i => RotationGate(operator, 0.))(num_qubits) for i in 1:num_qubits] |> chain
end

function coupling_gates(num_qubits::Int, operators=[Z, Z])
    return [put((i, j) => RotationGate(kron(operators...), 0.))(num_qubits) for j in 1:num_qubits for i in 1:j-1] |> chain
end

function objective_function(num_qubits::Int)
    [local_field_gates(num_qubits), coupling_gates(num_qubits)] |> chain
end

function driver(num_qubits::Int, G::YaoBlocks.PauliGate)
    local_field_gates(num_qubits, G)
end

function driver(num_qubits::Int, G::Vector{<:YaoBlocks.PauliGate})
    @assert size(G)[1] == 2
    coupling_gates(num_qubits, G)
end

function layer(problem::Problem)
    [objective_function(problem.num_qubits), driver(problem.num_qubits, problem.driver)] |> chain
end

Zygote.@nograd function circuit(problem::Problem)
    [layer(problem) for _ in 1:problem.num_layers] |> chain
end