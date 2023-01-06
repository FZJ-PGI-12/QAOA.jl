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


function cobyla_optimize(problem::Problem, beta::Vector{Float64}, gamma::Vector{Float64}; niter::Int=128)
    @unpack_Problem problem

    opt = Opt(:LN_COBYLA, 2num_layers)
    opt.lower_bounds = 0.
    opt.upper_bounds = pi .* vcat([1 for _ in 1:num_layers], [2 for _ in 1:num_layers])
    opt.maxeval = niter
    # opt.xtol_rel = 1e-5
    # opt.xtol_abs = 1e-5

    f = (x, _) -> cost_function(problem, x)

    opt.max_objective = f
    # opt.min_objective = f

    cost, params, info = optimize(opt, vcat(beta, gamma))

    circ = circuit(problem)
    circ = dispatch_parameters(circ, problem, params)    
    probabilities = uniform_state(num_qubits) |> circ |> probs
    cost, params, probabilities
end


# TEST MISSING
function gradient_optimize(problem::Problem, beta_and_gamma::Vector{Real}; niter::Int=128)
    @unpack_Problem problem

    f = x -> cost_function(problem, x)

    learning_rate = 0.05
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



