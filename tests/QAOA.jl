using Test, PyCall, Yao, Zygote
np = pyimport("numpy")
nx = pyimport("networkx")
include("./../src/QAOA.jl")


@testset "expectation value for constant beta and gamma" begin

    num_qubits = 4
    np.random.seed(1)
    a = np.random.rand(num_qubits)
    a = np.sort(a)

    @test a ≈ [0.00011437481734488664, 0.30233257263183977, 0.417022004702574, 0.7203244934421581] rtol=1e-10

    J = -2 * np.outer(a |> transpose, a)
    np.fill_diagonal(J, 0.)

    # freeze final spin to +1
    h_z = J[end, 1:end-1]
    J = J[1:end-1, 1:end-1]
    num_qubits += -1

    p = 3
    beta_and_gamma = (pi/4)*ones(2p)

    problem = QAOA.Problem(p, h_z, J)
    circ = QAOA.circuit(problem)

    circ = QAOA.dispatch_parameters(circ, problem, beta_and_gamma)

    reg = uniform_state(num_qubits) |> circ
    loss = (expect(QAOA.problem_hamiltonian(problem), reg) |> real) - sum(a.^2)

    @test loss ≈ -1.0165692236963693 rtol = 1e-10

end


@testset "number partitioning with standard annealing schedule" begin

    rtol = 1e-10

    num_qubits = 4
    np.random.seed(3)
    a = np.random.rand(num_qubits)
    a = np.sort(a)

    @test a ≈ [0.2909047389129443, 0.510827605197663, 0.5507979025745755, 0.7081478226181048] rtol=rtol

    J = -2 * np.outer(a |> transpose, a)
    np.fill_diagonal(J, 0.)

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

    circ = QAOA.dispatch_parameters(circ, problem, vcat(beta, gamma))

    reg = uniform_state(num_qubits) |> circ
    loss = (expect(QAOA.problem_hamiltonian(problem), reg) |> real) - sum(a.^2)

    @test loss ≈ -0.3323235141793123 rtol = rtol

    # cost, params, probabilities = QAOA.optimize_parameters(problem, vcat(beta, gamma), :LN_COBYLA; niter=200)

    # @test cost ≈ 0.9216008378625351 rtol = rtol
    # @test params ≈ [0.9728508285376499, 0.37387633456882235, 0.38551923541866506, 0.19184302643416146, 0.054492653127357515, 0.1287207004967745, 0.15464896912564985, 0.3326844430731712, 0.8185885963121214, 0.573225992828709] rtol = rtol
    # @test probabilities ≈ [8.209223151631255e-5, 0.007557979860814958, 0.018305942781999488, 0.20516668882823796, 0.030727741219340948, 0.2190891178408314, 0.30558203338494205, 0.21348840385231707] rtol = rtol

end

@testset "number partitioning with second-order annealing schedule" begin

    rtol = 1e-10

    num_qubits = 4
    np.random.seed(3)
    a = np.random.rand(num_qubits)
    a = np.sort(a)

    @test a ≈ [0.2909047389129443, 0.510827605197663, 0.5507979025745755, 0.7081478226181048] rtol=rtol

    J = -2 * np.outer(a |> transpose, a)
    np.fill_diagonal(J, 0.)

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

    circ = QAOA.dispatch_parameters(circ, problem, vcat(beta, gamma))

    reg = uniform_state(num_qubits) |> circ
    loss = (expect(QAOA.problem_hamiltonian(problem), reg) |> real) - sum(a.^2)

    @test loss ≈ -0.30937725031837593 rtol = rtol

    # cost, params, probabilities = QAOA.optimize_parameters(problem, vcat(beta, gamma), :LN_COBYLA; niter=200)

    # @test cost ≈ 1.0207507624809784 rtol = rtol
    # @test params ≈ [0.45552357346323913, 0.8519617534567095, 0.41672070806223344, 0.1671354206050005, 0.07979288769770952, 0.1375974898553044, 0.3904358072281066, 0.627426432889484, 1.0729567865537324, 1.7763875375849982] rtol = rtol
    # @test probabilities ≈ [0.00012918673824191518, 0.004260234783747874, 0.002767092222922995, 0.24777644431493934, 0.005042041028644788, 0.27990883903555636, 0.41455048678111517, 0.04556567509483212] rtol = rtol

end


@testset "analytic gradient" begin

    rtol = 1e-10

    num_qubits = 2

    h_z = zeros(num_qubits)
    J = np.array([[0, 1], [1, 0]]) # C = Z_1 * Z_2

    p = 1
    problem = QAOA.Problem(p, h_z, J)

    beta_vals = np.linspace(0, pi, 11)
    gamma_vals = np.linspace(0, pi, 11)
    X, Y = np.meshgrid(beta_vals, gamma_vals)

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