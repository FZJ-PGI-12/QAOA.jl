@testset "expectation value for constant beta and gamma" begin

    num_qubits = 4
    a = [0.00011437481734488664, 0.30233257263183977, 0.417022004702574, 0.7203244934421581]

    J = -2 .* (a * transpose(a))
    J[diagind(J)] .= 0.0    

    # freeze final spin to +1
    h_z = J[end, 1:end-1]
    J = J[1:end-1, 1:end-1]
    num_qubits += -1

    p = 3
    beta_and_gamma = (pi/4)*ones(2p)

    problem = QAOA.Problem(p, h_z, J)
    circ = QAOA.circuit(problem)

    circ = QAOA.dispatch_parameters!(circ, problem, beta_and_gamma)

    reg = uniform_state(num_qubits) |> circ
    loss = (expect(QAOA.problem_hamiltonian(problem), reg) |> real) - sum(a.^2)

    @test loss ≈ -1.0165692236963693 rtol = 1e-10

end


@testset "break Z2 symmetry" begin

    num_qubits = 4
    a = [0.00011437481734488664, 0.30233257263183977, 0.417022004702574, 0.7203244934421581]

    J = -2 .* (a * transpose(a))
    J[diagind(J)] .= 0.0        

    p = 3
    beta_and_gamma = (pi/4)*ones(2p)

    problem = QAOA.Problem(p, J)
    circ = QAOA.circuit(problem)

    circ = QAOA.dispatch_parameters!(circ, problem, beta_and_gamma)

    reg = uniform_state(nqubits(circ)) |> circ
    loss = (expect(QAOA.problem_hamiltonian(problem), reg) |> real) - sum(a.^2)

    @test loss ≈ -1.0165692236963693 rtol = 1e-10

end


@testset "number partitioning with standard annealing schedule" begin

    rtol = 1e-10

    num_qubits = 4
    a = [0.2909047389129443, 0.510827605197663, 0.5507979025745755, 0.7081478226181048]

    J = -2 .* (a * transpose(a))
    J[diagind(J)] .= 0.0        

    # freeze final spin to +1
    h_z = J[end, 1:end-1]
    J = J[1:end-1, 1:end-1]
    num_qubits += -1

    p = 5

    # annealing schedule as initial parameters
    T = 0.5
    gamma = [T * j / (p - 1) for j in 0:p-1]
    beta  = [T * (1 - j / (p - 1)) for j in 0:p-1]

    problem = QAOA.Problem(p, h_z, J)
    circ = QAOA.circuit(problem)

    circ = QAOA.dispatch_parameters!(circ, problem, vcat(beta, gamma))

    reg = uniform_state(num_qubits) |> circ
    loss = (expect(QAOA.problem_hamiltonian(problem), reg) |> real) - sum(a.^2)

    @test loss ≈ -0.3323235141793123 rtol = rtol

end

@testset "number partitioning with second-order annealing schedule" begin

    rtol = 1e-10

    num_qubits = 4
    a = [0.2909047389129443, 0.510827605197663, 0.5507979025745755, 0.7081478226181048]

    J = -2 .* (a * transpose(a))
    J[diagind(J)] .= 0.0        

    # freeze final spin to +1
    h_z = J[end, 1:end-1]
    J = J[1:end-1, 1:end-1]
    num_qubits += -1

    p = 5

    # for this schedule, see Appendix A of https://arxiv.org/pdf/1907.02359.pdf
    gamma = [0.5(j-1/2)/p  for j in 1:p]
    beta = vcat([0.5(1-j/p) for j in 1:p-1], [0.5/(4p)])

    problem = QAOA.Problem(p, h_z, J)
    circ = QAOA.circuit(problem)

    circ = QAOA.dispatch_parameters!(circ, problem, vcat(beta, gamma))

    reg = uniform_state(num_qubits) |> circ
    loss = (expect(QAOA.problem_hamiltonian(problem), reg) |> real) - sum(a.^2)

    @test loss ≈ -0.30937725031837593 rtol = rtol

end


@testset "analytic gradient" begin

    rtol = 1e-10

    num_qubits = 2

    h_z = zeros(num_qubits)
    J = [0 1; 1 0] # C = Z_1 * Z_2

    p = 1
    problem = QAOA.Problem(p, h_z, J)

    vals = LinRange(0, pi, 11)
    X = vals' .* ones(11)
    Y = ones(11)' .* vals

    analytic_cost = x -> sin(4x[1])sin(2x[2])

    QAOA_gradient = (x, y) -> gradient(z -> QAOA.cost_function(problem, z), [x, y])[1]
    analytic_gradient = (x, y) -> gradient(analytic_cost, [x, y])[1]

    @test ((x, y) -> QAOA.cost_function(problem, [x, y])).(X, Y) ≈ ((x, y) -> analytic_cost([x, y])).(X, Y) rtol=rtol
    @test QAOA_gradient.(X, Y) ≈ analytic_gradient.(X, Y) rtol=rtol

end


@testset "minimum vertex cover from Pennylane" begin
    # https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html

    p = 2
    num_qubits = 4
    
    h = -1 .* [0.5, 0.5, 1.25, -0.25]
    J = -1 .* [0.0 0.75 0.75 0.0; 0.0 0.0 0.75 0.0; 0.0 0.0 0.0 0.75; 0.0 0.0 0.0 0.0]

    problem = QAOA.Problem(p, h, J)

    # test gradient optimization
    learning_rate = 0.01
    cost, params, probs = QAOA.optimize_parameters(problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)

    @test cost ≈ 1.9649341584194655 rtol = 1e-8
    @test params ≈ [0.48615868, 0.28701741, 0.28212724, 0.674492] rtol = 1e-4
    @test probs[6] ≈ 0.278 rtol = 1e-2
    @test probs[7] ≈ 0.278 rtol = 1e-2

    # test NLopt
    cost, params, probs = QAOA.optimize_parameters(problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA) 
    
    @test cost ≈ 1.9649341584194655 rtol = 1e-5
    @test params ≈ [0.48615868, 0.28701741, 0.28212724, 0.674492] rtol = 1e-2
    @test probs[6] ≈ 0.278 rtol = 1e-2
    @test probs[7] ≈ 0.278 rtol = 1e-2    
end