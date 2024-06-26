"""
    fluctuation_matrix(problem::Problem, S::Vector{<:Vector{<:Real}}, solutions::Vector{<:Real}, β::Real, γ::Real)

Returns the Gaussian fluctuation matrix for a given point along the mean-field trajectories. 
"""
function fluctuation_matrix(problem::Problem, S::Vector{<:Vector{<:Real}}, solutions::Vector{<:Real}, β::Real, γ::Real)
    @unpack_Problem problem
    
    A = zeros(ComplexF64, (num_qubits, num_qubits))
    B = zeros(ComplexF64, (num_qubits, num_qubits))
    τ_3 = diagm(vcat(ones(num_qubits), -ones(num_qubits)))

    # helper function to construct A and B    
    n_ij_pm = (idx, pm) -> solutions[idx] * S[idx][1] + pm * 1im * S[idx][2]

    for i in 1:num_qubits
        # we exclude a factor of 2 here because we symmetrize below
        # signs in front of β, γ are reversed relative to the original paper
        A[i, i] = -β * S[i][1] / (1 + solutions[i] * S[i][3]) - γ * solutions[i] * magnetization(S, local_fields, couplings)[i]
        for j in i + 1:num_qubits
            A[i, j] = γ * couplings[i, j] * n_ij_pm(i, 1) * n_ij_pm(j, -1)
            B[i, j] = γ * couplings[i, j] * n_ij_pm(i, 1) * n_ij_pm(j,  1)        
        end
    end

    # symmetrize
    A += transpose(conj.(A))
    B += transpose(B)

    L = τ_3 * [A                   B;
               transpose(conj.(B)) conj.(A)]

    L
end


"""
    evolve_fluctuations(problem::Problem, τ::Real, β::Vector{<:Real}, γ::Vector{<:Real})

### Input
- `problem::Problem`: A `Problem` instance defining the optimization problem.
- `τ::Real`: The time-step of the considered annealing schedule.
- `β::Vector{<:Real}`: The corresponding vector of QAOA driver parameters.
- `γ::Vector{<:Real}`: The corresponding vector of QAOA problem parameters.    

### Output
- The Lyapunov exponents characterizing the dynamics of the Gaussian fluctuations.
"""
function evolve_fluctuations(problem::Problem, τ::Real, β::Vector{<:Real}, γ::Vector{<:Real})
    @unpack_Problem problem
    
    @assert size(β)[1] == size(γ)[1] "Invalid QAOA parameters β and γ!"

    lyapunov_exponent = [zeros(2num_qubits) for _ in 1:num_layers]
    
    # evolution
    S = [[1., 0., 0.] for _ in 1:num_qubits]
    S = [S for _ in 1:num_layers+1]
    evolve!(S, local_fields, couplings, β, γ)
    solutions = mean_field_solution(S[end])  
    
    M = 1.0I(2num_qubits)
    for k in 1:size(γ)[1]
        L = fluctuation_matrix(problem, S[k], solutions, β[k], γ[k])        
        M = exp(-1im .* τ .* L) * M

        lyapunov_exponential_eig = eigvals(M * transpose(conj.(M)))
        lyapunov_exponent[k] = log.((1.0 + 0.0im) * lyapunov_exponential_eig) ./ 2 .|> real |> sort        
    end

    lyapunov_exponent
end