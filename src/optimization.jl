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


Zygote.@nograd function problem_hamiltonian(problem::Problem)
    H =  sum([problem.local_fields[i] * put(i => Z)(problem.num_qubits) for i in 1:problem.num_qubits])
    H += sum([problem.couplings[i, j] * put((i, j) => kron(Z, Z))(problem.num_qubits) for j in 1:problem.num_qubits for i in 1:j-1])
    H
end

function cost_function(problem::Problem, beta_and_gamma)::Real
    @unpack_Problem problem
    circ = circuit(problem)
    circ = dispatch_parameters(circ, problem, beta_and_gamma)
    reg = apply(uniform_state(nqubits(circ)), circ)
    expect(QAOA.problem_hamiltonian(problem), reg) |> real
end


function optimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}, algorithm; niter::Int=128)
    @unpack_Problem problem

    opt = Opt(algorithm, 2num_layers)
    opt.lower_bounds = 0.
    opt.upper_bounds = pi .* vcat([1 for _ in 1:num_layers], [2 for _ in 1:num_layers])
    opt.maxeval = niter
    # opt.xtol_rel = 1e-5
    # opt.xtol_abs = 1e-5

    f = (x, _) -> cost_function(problem, x)

    opt.max_objective = f
    # opt.min_objective = f

    cost, params, info = optimize(opt, beta_and_gamma)

    circ = circuit(problem)
    circ = dispatch_parameters(circ, problem, params)
    probabilities = uniform_state(num_qubits) |> circ |> probs
    cost, params, probabilities
end


function optimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}; niter::Int=128, learning_rate::Float64 = 0.05)
    @unpack_Problem problem

    f = x -> cost_function(problem, x)

    learning_rate = learning_rate
    cost = f(beta_and_gamma)
    params = beta_and_gamma

    for n in 1:niter
        params = params .+ learning_rate .* gradient(f, params)[1]
        cost = f(params)
    end

    circ = circuit(problem)
    circ = dispatch_parameters(circ, problem, params)
    probabilities = uniform_state(num_qubits) |> circ |> probs
    cost, params, probabilities
end



