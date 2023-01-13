function local_field_gates(num_qubits::Int, gate::Function=Rz)
    return [put(i => gate(0.))(num_qubits) for i in 1:num_qubits] |> chain
end

function coupling_gates(num_qubits::Int, operators=[Z, Z])
    return [put((i, j) => RotationGate(kron(operators...), 0.))(num_qubits) for j in 1:num_qubits for i in 1:j-1] |> chain
end

function objective_function(num_qubits::Int)
    [local_field_gates(num_qubits), coupling_gates(num_qubits)] |> chain
end

function layer(num_qubits::Int)
    [objective_function(num_qubits), local_field_gates(num_qubits, Rx)] |> chain
end

Zygote.@nograd function circuit(problem::Problem)
    [layer(problem.num_qubits) for _ in 1:problem.num_layers] |> chain
end