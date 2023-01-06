Zygote.@nograd function problem_parameters(local_fields::Vector{Real}, couplings::Matrix{Real})::Vector{Real}
    num_qubits = size(local_fields)[1]
    vcat(2 .* local_fields, [2 * couplings[i, j] for j in 1:num_qubits for i in 1:j-1])
end


function dispatch_parameters(circ, problem::Problem, beta_and_gamma)
    @unpack_Problem problem
    circ = dispatch(circ, reduce(vcat,
                                    [vcat(beta_and_gamma[l + num_layers] .* problem_parameters(local_fields, couplings),
                                          beta_and_gamma[l] .* 2. .* ones(num_qubits)
                                         )
                                    for l in 1:num_layers]
                                )
                   )
    circ
end
