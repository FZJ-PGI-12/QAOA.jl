var documenterSearchIndex = {"docs":
[{"location":"examples/mean_field/#MFAOA","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"","category":"section"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"tip: Tip\nFor more details on the mean-field Approximate Optimization Algorithm, please consult our paper.","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"note: Note\nA Jupyter notebook related to this example is available in our examples folder. For a comparison between the QAOA and the mean-field AOA, have a look into our MaxCut example.","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"In close analogy to the QAOA, the mean-field Hamiltonian reads","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"beginalign\n    H(t) =  gamma(t)sum_i=1^N bigg h_i + sum_ji J_ij  n_j^z(t) bigg n_i^z(t) + beta(t) sum_i=1^N n_i^x(t)\nendalign","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"The mean-field evolution is then given by","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"beginalign\n    boldsymboln_i(p) = prod_k=1^p hat V_i^D(k) hat V_i^P(k) boldsymboln_i(0)\nendalign","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"where the initial spin vectors are boldsymboln_i(0) = (1 0 0)^T  forall i, and the rotation matrices hat V_i^DP are defined as","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"beginalign\nhat V_i^D(k) = \nbeginpmatrix\n1  0  0 \n0  cos(2 Delta_ibeta_k)  -sin(2 Delta_i beta_k) \n0  sin (2 Delta_i beta_k)  phantom-cos(2 Delta_i beta_k) \nendpmatrix\nendalign","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"and","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"beginalign\nhat V_i^P(k) = \nbeginpmatrix\ncos(2m_i (t_k-1) gamma_k)  -sin(2m_i (t_k-1) gamma_k)  0 \nsin (2m_i (t_k-1) gamma_k)  phantom-cos(2m_i (t_k-1) gamma_k)  0 \n0  0  1\nendpmatrix\nendalign","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"with the magnetization ","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"beginalign\nm_i(t) = h_i + sum_j=1^N J_ij n_j^z(t)\nendalign","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"To implement these dynamics within QAOA.jl, we begin by defining a schedule:","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"using QAOA, LinearAlgebra\nimport Random, Distributions\n\np = 100\nτ = 0.5\nγ = τ .* ((1:p) .- 1/2) ./ p |> collect\nβ = τ .* (1 .- (1:p) ./ p) |> collect\nβ[p] = τ / (4 * p)","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"Next, we set up a random instance of the Sherrington-Kirkpatrick model:","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"N = 5\nσ2 = 1.0\n\nRandom.seed!(1)\nJ = rand(Distributions.Normal(0, σ2), N, N) ./ sqrt(N) \nJ[diagind(J)] .= 0.0\nJ = UpperTriangular(J)\nJ = J + transpose(J)","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"Then we are ready to construct a Problem:","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"mf_problem = Problem(p, J)","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"note: Note\nWhen the constructor for Problem is called without local_fields, then QAOA.jl will automatically break the mathcalZ_2 symmetry of the underlying Hamiltonian. That is, the final spin will be held fixed, which automatically introduces effective local_fields. While this is not required for the QAOA, it is a necessary preliminary for the mean-field AOA, since the mean-field dynamics will otherwise be trivial. ","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"This is all we need to call the mean-field dynamics. The initial values are","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"S = [[1., 0., 0.] for _ in 1:N-1]","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"where we have taken into account that the final spin is fixed. The final vector of spins is then obtained as","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"S = evolve!(S, mf_problem.local_fields, mf_problem.couplings, β, γ)","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"The energy expectation value in mean-field approximation is ","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"E = expectation(S[end], mf_problem.local_fields, mf_problem.couplings)","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"and the solution of the algorithm can be retrieved by calling","category":"page"},{"location":"examples/mean_field/","page":"Mean-Field Approximate Optimization Algorithm","title":"Mean-Field Approximate Optimization Algorithm","text":"sol = mean_field_solution(S[end])","category":"page"},{"location":"examples/min_vertex_cover/#MinVertexCover","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"","category":"section"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"note: Note\nA Jupyter notebook related to this example is available in our examples folder. See also Wikipedia.","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"To be able to directly compare to the Pennylane implementation, we employ the following cost function:","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"beginalign*\n    hat C = -frac 34 sum_(i j) in E(G) (hat Z_i hat Z_j  +  hat Z_i  +  hat Z_j)  + sum_i in V(G) hat Z_i\nendalign*","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"where E(G) is the set of edges and V(G) is the set of vertices of the graph G (we put a global minus sign since we maximize the cost function).","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"We can set this model up as follows:","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"using QAOA, LinearAlgebra\nimport Random, Distributions\nusing PyCall\nnx = pyimport(\"networkx\")\n\nN = 4\ngraph = nx.gnp_random_graph(N, 0.5, seed=7) \n\nh = -ones(N)\nJ = zeros(N, N)\nfor edge in graph.edges\n    h[edge[1] + 1] += 3/4.\n    h[edge[2] + 1] += 3/4.\n    J[(edge .+ (1, 1))...] = 3/4.\nend","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"We have two options to get the corresponding Problem from QAOA.jl. We can pass the coupling matrix J directly:","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"p = 2\nmvc_problem = QAOA.Problem(p, -h, -J)","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"(since our algorithm maximizes the cost function, we put in extra minus signs for the problem parameters), or we can use a predefined wrapper function that constructs J from the above parameters and directly returns a Problem:","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"mvc_problem = QAOA.min_vertex_cover(N, [edge .+ (1, 1) for edge in graph.edges], num_layers=p)","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"Given mvc_problem, we can then call the gradient optimizer:","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"learning_rate = 0.01\ncost, params, probs = QAOA.optimize_parameters(mvc_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"Alternatively, we can use NLsolve.jl:","category":"page"},{"location":"examples/min_vertex_cover/","page":"Minimum Vertex Cover","title":"Minimum Vertex Cover","text":"cost, params, probs = QAOA.optimize_parameters(mvc_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)","category":"page"},{"location":"citation/#Citation","page":"Citation","title":"Citation","text":"","category":"section"},{"location":"citation/","page":"Citation","title":"Citation","text":"If you are using QAOA.jl, please cite our work:","category":"page"},{"location":"citation/","page":"Citation","title":"Citation","text":"@misc{https://doi.org/10.48550/arxiv.2303.00329,\n  doi = {10.48550/ARXIV.2303.00329},\n  url = {https://arxiv.org/abs/2303.00329},\n  author = {Misra-Spieldenner, Aditi and Bode, Tim and Schuhmacher, Peter K. and Stollenwerk, Tobias and Bagrets, Dmitry and Wilhelm, Frank K.},\n  keywords = {Quantum Physics (quant-ph), Disordered Systems and Neural Networks (cond-mat.dis-nn), FOS: Physical sciences, FOS: Physical sciences},\n  title = {Mean-Field Approximate Optimization Algorithm},\n  publisher = {arXiv},\n  year = {2023},\n  copyright = {arXiv.org perpetual, non-exclusive license}\n}","category":"page"},{"location":"examples/max_cut/#MaxCut","page":"MaxCut","title":"MaxCut","text":"","category":"section"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"note: Note\nA Jupyter notebook related to this example is available in our examples folder.","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"The cost function for the MaxCut problem as defined in the original QAOA paper is","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"beginalign*\nhat C = frac 12 sum_(i j) in E(G) (1 - hat Z_i hat Z_j)\nendalign*","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"where E(G) is the set of edges of the graph G. ","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"We can set this model up as follows:","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"using QAOA, LinearAlgebra\nimport Random, Distributions\nusing PyCall\nnx = pyimport(\"networkx\");\n\nN = 4\ngraph = nx.cycle_graph(N) \n\nh = zeros(N)\nJ = zeros(N, N)\nfor edge in graph.edges\n    J[(edge .+ (1, 1))...] = -1/2.\nend","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"Note that we have to shift the edges by 1 when going from Python to Julia. We have two options to get the corresponding Problem from QAOA.jl. We can pass the coupling matrix J directly:","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"p = 1\nmax_cut_problem = QAOA.Problem(p, h, J)","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"or we can use a predefined wrapper function that constructs J from the above parameters and directly returns a Problem:","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"max_cut_problem = QAOA.max_cut(N, [edge .+ (1, 1) for edge in graph.edges], num_layers=p)","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"Given max_cut_problem, we can then call the gradient optimizer:","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"learning_rate = 0.01\ncost, params, probs = QAOA.optimize_parameters(max_cut_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"Alternatively, we can use NLsolve.jl:","category":"page"},{"location":"examples/max_cut/","page":"MaxCut","title":"MaxCut","text":"cost, params, probs = QAOA.optimize_parameters(max_cut_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)","category":"page"},{"location":"examples/partition_problem/#PartitionProblem","page":"Partition Problem","title":"Partition Problem","text":"","category":"section"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"note: Note\nA Jupyter notebook related to this example is available in our examples folder.","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"The partition problem (see also Wikipedia) for a set of uniformly distributed numbers mathcalS = a_1  a_N consists of finding two subsets mathcalS_1 cup mathcalS_2 =  mathcalS such that the difference of the sums over the two subsets mathcalS_1 2 is as small as possible. The cost function in Ising form can be defined as ","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"beginalign*\nhat C = -left(sum_i=1^N a_i hatZ_iright)^2 = sum_ijleq N J_ij hatZ_i hatZ_j + mathrmconst\nendalign*","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"with J_ij=-2a_i a_j. The goal is then to maximize hat C.","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"We can set this model up as follows:","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"using QAOA, LinearAlgebra\nimport Random, Distributions\n\nN = 4\nRandom.seed!(1)\na = rand(Distributions.Uniform(0, 1), N)\n\nJ = -2 .* (a * transpose(a))\nJ[diagind(J)] .= 0.0","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"We have two options to get the corresponding Problem from QAOA.jl. We can pass the coupling matrix J directly:","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"p = 4\npartition_problem = QAOA.Problem(p, zeros(N), J)","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"or we can use a predefined wrapper function that constructs J from the above parameters and directly returns a Problem:","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"partition_problem = QAOA.partition_problem(a, num_layers=p)","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"Given partition_problem, we can then call the gradient optimizer:","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"learning_rate = 0.05\ncost, params, probs = QAOA.optimize_parameters(partition_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"Alternatively, we can use NLsolve.jl:","category":"page"},{"location":"examples/partition_problem/","page":"Partition Problem","title":"Partition Problem","text":"cost, params, probs = QAOA.optimize_parameters(partition_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)","category":"page"},{"location":"#Welcome!","page":"Overview","title":"Welcome!","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"QAOA.jl is a lightweight implementation of the Quantum Approximate Optimization Algorithm (QAOA) based on Yao.jl. Furthermore, it implements the mean-field Approximate Optimization Algorithm, which is a useful tool to simulate quantum annealing and the QAOA in the mean-field approximation.","category":"page"},{"location":"#Installation","page":"Overview","title":"Installation","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"To install, use Julia's built-in package manager","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"julia> ] add QAOA","category":"page"},{"location":"#Library","page":"Overview","title":"Library","text":"","category":"section"},{"location":"#Index","page":"Overview","title":"Index","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"","category":"page"},{"location":"#Problem-Structure","page":"Overview","title":"Problem Structure","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"Problem{}","category":"page"},{"location":"#QAOA.Problem","page":"Overview","title":"QAOA.Problem","text":"Problem(num_layers::Int, local_fields::Vector{Real}, couplings::Matrix{Real}, driver)\n\nA container holding the basic properties of the QAOA circuit.\n\nnum_qubits::Int64: The number of qubits of the QAOA circuit.\nnum_layers::Int64: The number of layers p of the QAOA circuit.\nlocal_fields::Vector{Real}: The local (magnetic) fields of the Ising problem Hamiltonian.\ncouplings::Matrix{Real}: The coupling matrix of the Ising problem Hamiltonian.\nedges::Any: The edges of the graph specified by the coupling matrix.\ndriver::Any: The driver of the QAOA circuit. By default the Pauli matrix X. May also be set to,     e.g., [[X, X], [Y, Y]] to obtain the drivershat X_i hat X_j + hat Y_i hat Y_j acting on every pair of connected qubits.\n\n\n\n\n\n","category":"type"},{"location":"#Cost-Function-and-QAOA-Parameter-Optimization","page":"Overview","title":"Cost Function and QAOA Parameter Optimization","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"cost_function(problem::Problem, beta_and_gamma::Vector{Float64})\noptimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}, algorithm; niter::Int=128)\noptimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}; niter::Int=128, learning_rate::Float64 = 0.05)","category":"page"},{"location":"#QAOA.cost_function-Tuple{Problem, Vector{Float64}}","page":"Overview","title":"QAOA.cost_function","text":"cost_function(problem::Problem, beta_and_gamma::Vector{Float64})\n\nReturns the QAOA cost function, i.e. the expectation value of the problem Hamiltonian.\n\nInput\n\nproblem::Problem: A Problem instance defining the optimization problem.\nbeta_and_gamma::Vector{Float64}: A vector of QAOA parameters.\n\nOutput\n\nThe expectation value of the problem Hamiltonian.\n\nNotes\n\nThis function computes\n\nlangle hat H rangle = leftlangle sum_i=1^Nleft( h_i hat Z_i + sum_ji J_ijhat Z_i hat Z_j right)rightrangle\n\nwhere N is num_qubits, h_i are the local_fields and J_ij are the couplings from problem.\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.optimize_parameters-Tuple{Problem, Vector{Float64}, Any}","page":"Overview","title":"QAOA.optimize_parameters","text":"optimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}, algorithm; niter::Int=128)\n\nGradient-free optimization of the QAOA cost function using NLopt.\n\nInput\n\nproblem::Problem: A Problem instance defining the optimization problem.\nbeta_and_gamma::Vector{Float64}: The vector of initial QAOA parameters.\nalgorithm: One of NLopt's algorithms.\nniter::Int=128 (optional): Number of optimization steps to be performed.\n\nOutput\n\ncost: Final value of the cost function.\nparams: The optimized parameters.\nprobabilities: The simulated probabilities of all possible outcomes.\n\nExample\n\nFor given number of layers p, local fields h and couplings J, define the problem \n\nproblem = QAOA.Problem(p, h, J) \n\nand then do\n\ncost, params, probs = QAOA.optimize_parameters(problem, ones(2p), :LN_COBYLA).\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.optimize_parameters-Tuple{Problem, Vector{Float64}}","page":"Overview","title":"QAOA.optimize_parameters","text":"optimize_parameters(problem::Problem, beta_and_gamma::Vector{Float64}; niter::Int=128, learning_rate::Float64 = 0.05)\n\nGradient optimization of the QAOA cost function using Zygote.\n\nInput\n\nproblem::Problem: A Problem instance defining the optimization problem.\nbeta_and_gamma::Vector{Float64}: The vector of initial QAOA parameters.\nniter::Int=128 (optional): Number of optimization steps to be performed.\nlearning_rate::Float64=0.05 (optional): The learning rate of the gradient-ascent method.\n\nOutput\n\ncost: Final value of the cost function.\nparams: The optimized parameters.\nprobabilities: The simulated probabilities of all possible outcomes.\n\nNotes\n\nThe gradient-ascent method is defined via the parameter update\n\nparams = params .+ learning_rate .* gradient(f, params)[1].\n\nExample\n\nFor given number of layers p, local fields h and couplings J, define the problem \n\nproblem = QAOA.Problem(p, h, J) \n\nand then do\n\ncost, params, probs = QAOA.optimize_parameters(problem, ones(2p)).\n\n\n\n\n\n","category":"method"},{"location":"#Mean-Field-Approximate-Optimization-Algorithm","page":"Overview","title":"Mean-Field Approximate Optimization Algorithm","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"evolve!(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})\nevolve!(S::Vector{<:Vector{<:Vector{<:Real}}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})\nexpectation(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})\nmean_field_solution(problem::Problem, β::Vector{<:Real}, γ::Vector{<:Real})\nmean_field_solution(S::Vector{<:Vector{<:Real}})","category":"page"},{"location":"#QAOA.evolve!-Tuple{Vector{<:Vector{<:Real}}, Vector{<:Real}, Matrix{<:Real}, Vector{<:Real}, Vector{<:Real}}","page":"Overview","title":"QAOA.evolve!","text":"evolve!(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})\n\nInput\n\nS::Vector{<:Vector{<:Real}}: The initial vector of all spin vectors.\nh::Vector{<:Real}: The vector of local magnetic fields of the problem Hamiltonian.\nJ::Matrix{<:Real}: The coupling matrix  of the problem Hamiltonian.\nβ::Vector{<:Real}: The vector of QAOA driver parameters.\nγ::Vector{<:Real}: The vector of QAOA problem parameters.\n\nOutput\n\nThe vector of all spin vectors after a full mean-field AOA evolution.\n\nNotes\n\nThis is the first dispatch of evolve, which only returns the final vector of spin vectors.\nThe vector of spin vectors is properly initialized as\nS = [[1., 0., 0.] for _ in 1:num_qubits].\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.evolve!-Tuple{Vector{<:Vector{<:Vector{<:Real}}}, Vector{<:Real}, Matrix{<:Real}, Vector{<:Real}, Vector{<:Real}}","page":"Overview","title":"QAOA.evolve!","text":"evolve!(S::Vector{<:Vector{<:Vector{<:Real}}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})\n\nInput\n\nS::Vector{<:Vector{<:Vector{<:Real}}}: An empty history of the vector of all spin vectors.\nh::Vector{<:Real}: The vector of local magnetic fields of the problem Hamiltonian.\nJ::Matrix{<:Real}: The coupling matrix  of the problem Hamiltonian.\nβ::Vector{<:Real}: The vector of QAOA driver parameters.\nγ::Vector{<:Real}: The vector of QAOA problem parameters.\n\nOutput\n\nThe full history of the vector of all spin vectors after a full mean-field AOA evolution.\n\nNotes\n\nThis is the second dispatch of evolve, which returns the full history of the vector of spin vectors.    \nFor a schedule of size num_layers = size(β)[1], S can be initialized as \nS = [[[1., 0., 0.] for _ in 1:num_qubits] for _ in 1:num_layers+1].\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.expectation-Tuple{Vector{<:Vector{<:Real}}, Vector{<:Real}, Matrix{<:Real}}","page":"Overview","title":"QAOA.expectation","text":"expectation(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})\n\nInput\n\nS::Vector{<:Vector{<:Real}}: A vector of all spin vectors.\nh::Vector{<:Real}: A vector of local magnetic fields.\nJ::Matrix{<:Real}: A coupling matrx.\n\nOutput\n\nThe energy expectation value corresponding to the supplied spin configuration.\n\nNotes\n\nIn the mean-field approximation, the energy expectation value is defined as\nlangle E rangle = - sum_i=1^N bigg h_i + sum_ji J_ij  n_j^z(p) bigg n_i^z(p)\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.mean_field_solution-Tuple{Problem, Vector{<:Real}, Vector{<:Real}}","page":"Overview","title":"QAOA.mean_field_solution","text":"mean_field_solution(problem::Problem, β::Vector{<:Real}, γ::Vector{<:Real})\n\nInput\n\nproblem::Problem: A Problem instance defining the optimization problem.\nβ::Vector{<:Real}: The vector of QAOA driver parameters.\nγ::Vector{<:Real}: The vector of QAOA problem parameters.\n\nOutput\n\nThe solution bitstring computed for a given problem and schedule parameters.\n\nNotes\n\nThis is the first dispatch of mean_field_solution, which computes the sought-for solution bitstring for a given problem and schedule parameters.\nThe solution bitstring boldsymbolsigma_* is defined as follows in terms of the z components of the final spin vectors:\nboldsymbolsigma_* = left(mathrmsign(n_1^z)  mathrmsign(n_N^z) right)\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.mean_field_solution-Tuple{Vector{<:Vector{<:Real}}}","page":"Overview","title":"QAOA.mean_field_solution","text":"mean_field_solution(S::Vector{<:Vector{<:Real}})\n\nInput\n\nproblem::Problem: A Problem instance defining the optimization problem.\nβ::Vector{<:Real}: The vector of QAOA driver parameters.\nγ::Vector{<:Real}: The vector of QAOA problem parameters.\n\nOutput\n\nThe solution bitstring computed from a given vector of spin vectors.\n\nNotes\n\nThis is the second dispatch of mean_field_solution, which computes the solution bitstring from a given vector of spin vectors.\nThe solution bitstring boldsymbolsigma_* is defined as follows in terms of the z components of the spin vectors:\nboldsymbolsigma_* = left(mathrmsign(n_1^z)  mathrmsign(n_N^z) right)\n\n\n\n\n\n","category":"method"},{"location":"#Fluctuation-Analysis","page":"Overview","title":"Fluctuation Analysis","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"evolve_fluctuations(problem::Problem, τ::Real, β::Vector{<:Real}, γ::Vector{<:Real})","category":"page"},{"location":"#QAOA.evolve_fluctuations-Tuple{Problem, Real, Vector{<:Real}, Vector{<:Real}}","page":"Overview","title":"QAOA.evolve_fluctuations","text":"evolve_fluctuations(problem::Problem, τ::Real, β::Vector{<:Real}, γ::Vector{<:Real})\n\nInput\n\nproblem::Problem: A Problem instance defining the optimization problem.\nτ::Real: The time-step of the considered annealing schedule.\nβ::Vector{<:Real}: The corresponding vector of QAOA driver parameters.\nγ::Vector{<:Real}: The corresponding vector of QAOA problem parameters.    \n\nOutput\n\nThe Lyapunov exponents characterizing the dynamics of the Gaussian fluctuations.\n\n\n\n\n\n","category":"method"},{"location":"#Predefined-Optimization-Problems","page":"Overview","title":"Predefined Optimization Problems","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"sherrington_kirkpatrick(N::Int, variance::Float64; seed::Int=1, num_layers::Int=1, driver=X)\npartition_problem(a::Vector{Float64}; num_layers::Int=1, driver=X)\nmax_cut(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)\nmin_vertex_cover(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)","category":"page"},{"location":"#QAOA.sherrington_kirkpatrick-Tuple{Int64, Float64}","page":"Overview","title":"QAOA.sherrington_kirkpatrick","text":"sherrington_kirkpatrick(variance::Float64; seed::Float64=1.0, num_layers::Int=1, driver=X)\n\nWrapper function setting up an instance of the Sherrington-Kirkpatrick model.\n\nInput\n\nN::Int: The number of spins of the problem.\nvariance::Float64: The variance of the distribution of the coupling matrix.\nseed::Float64=1.0: The seed for the random-number generator used in the coupling matrix.\nnum_layers::Int=1 (optional): The number of QAOA layers usually denoted by p.\ndriver=X (optional): The driver or mixer used in the QAOA.\n\nOutput\n\nAn instance of the Problem struct holding all relevant quantities.\n\nNotes\n\nThe cost function of the Sherrington-Kirkpatrick model is\n\nhat H_P = frac1sqrtNsum_ijleq N J_ij hatZ_i hatZ_j\n\nwhere the couplings J_ij are i.i.d. standard Gaussian variables,  i.e. with zero mean langle J_ij rangle = 0 and variance langle J_ij^2 rangle = J^2.\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.partition_problem-Tuple{Vector{Float64}}","page":"Overview","title":"QAOA.partition_problem","text":"partition_problem(a::Vector{Float64}; num_layers::Int=1, driver=X)\n\nWrapper function setting up an instance of the partition problem.\n\nInput\n\na::Vector{Float64}: The input vector of numbers to be partitioned.\nnum_layers::Int=1 (optional): The number of QAOA layers usually denoted by p.\ndriver=X (optional): The driver or mixer used in the QAOA.\n\nOutput\n\nAn instance of the Problem struct holding all relevant quantities.\n\nNotes\n\nThe partition problem for a set of uniformly distributed numbers mathcalS = a_1  a_N  consists of finding two subsets mathcalS_1 cup mathcalS_2 =  mathcalS  such that the difference of the sums over the two subsets mathcalS_1 2 is as small as possible.  The cost function in Ising form can be defined as \n\nhat C = -left(sum_i=1^N a_i hatZ_iright)^2 = sum_ijleq N J_ij hatZ_i hatZ_j + mathrmconst\n\nwith J_ij=-2a_i a_j. The goal is then to maximize hat C.\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.max_cut-Tuple{Int64, Vector{Tuple{Int64, Int64}}}","page":"Overview","title":"QAOA.max_cut","text":"max_cut(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)\n\nWrapper function setting up an instance of the MaxCut problem for the graph graph.\n\nInput\n\ngraph::PyObject: The input graph, must be a Python NetworkX graph.\nnum_layers::Int=1 (optional): The number of QAOA layers usually denoted by p.\ndriver=X (optional): The driver or mixer used in the QAOA.\n\nOutput\n\nAn instance of the Problem struct holding all relevant quantities.\n\nNotes\n\nThe cost function for the MaxCut problem as defined in the original QAOA paper is\n\nhat C = -frac12 sum_(i j) in E(G) hat Z_i hat Z_j + mathrmconst\n\nwhere E(G) is the set of edges of the graph G.\n\n\n\n\n\n","category":"method"},{"location":"#QAOA.min_vertex_cover-Tuple{Int64, Vector{Tuple{Int64, Int64}}}","page":"Overview","title":"QAOA.min_vertex_cover","text":"min_vertex_cover(num_nodes::Int, edges::Vector{Tuple{Int, Int}}; num_layers::Int=1, driver=X)\n\nWrapper function setting up a problem instance for the minimum vertex cover of the graph graph.\n\nInput\n\ngraph::PyObject: The input graph, must be a Python NetworkX graph.\nnum_layers::Int=1 (optional): The number of QAOA layers usually denoted by p.\ndriver=X (optional): The driver or mixer used in the QAOA.\n\nOutput\n\nAn instance of the Problem struct holding all relevant quantities.\n\nNotes\n\nThe cost function for the minimum-vertex-cover problem is \n\nhat C = -frac34 sum_(i j) in E(G) (hat Z_i hat Z_j  +  hat Z_i  +  hat Z_j)  + sum_i in V(G) hat Z_i\n\nwhere E(G) is the set of edges and V(G) is the set of vertices of graph (we have a global minus sign since we maximize the cost function).\n\n\n\n\n\n","category":"method"},{"location":"examples/sherrington_kirkpatrick/#SKModel","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"","category":"section"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"note: Note\nA Jupyter notebook related to this example is available in our examples folder.","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"The cost function of the SK model is defined as","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"beginalign*\nhat H_P = frac1sqrtNsum_ijleq N J_ij hatZ_i hatZ_j\nendalign*","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"where the couplings J_ij are i.i.d. standard Gaussian variables, i.e. with zero mean leftlangle J_ij rightrangle = 0 and variance leftlangle J_ij^2 rightrangle = J^2.","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"We can set this model up as follows:","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"using QAOA, LinearAlgebra\nimport Random, Distributions\n\nN = 4\nσ2 = 1.0\n\nRandom.seed!(1)\nJ = rand(Distributions.Normal(0, σ2), N, N) ./ sqrt(N) \nJ[diagind(J)] .= 0.0\nJ = UpperTriangular(J)\nJ = J + transpose(J)","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"We have two options to get the corresponding Problem from QAOA.jl. We can pass the coupling matrix J directly:","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"p = 2\nSK_problem = QAOA.Problem(p, zeros(N), J)","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"or we can use a predefined wrapper function that constructs J from the above parameters and directly returns a Problem:","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"SK_problem = QAOA.sherrington_kirkpatrick(N, σ2, num_layers=p, seed=1)","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"Given SK_problem, we can then call the gradient optimizer:","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"learning_rate = 0.02\ncost, params, probs = QAOA.optimize_parameters(SK_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]); learning_rate=learning_rate)","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"Alternatively, we can use NLsolve.jl:","category":"page"},{"location":"examples/sherrington_kirkpatrick/","page":"Sherrignton-Kirkpatrick Model","title":"Sherrignton-Kirkpatrick Model","text":"cost, params, probs = QAOA.optimize_parameters(SK_problem, vcat([0.5 for _ in 1:p], [0.5 for _ in 1:p]), :LN_COBYLA)","category":"page"}]
}
