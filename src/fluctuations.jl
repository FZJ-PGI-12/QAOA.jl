function evolve_fluctuations(problem::Problem, τ::Real, β::Vector{<:Real}, γ::Vector{<:Real})
    @unpack_Problem problem
    
    @assert size(β)[1] == size(γ)[1] "Invalid QAOA parameters β and γ!"
    
    # evolution
    S = [[1., 0., 0.] for _ in 1:num_qubits]
    S = [S for _ in 1:num_layers+1]
    S = evolve(S, local_fields, couplings, β, γ)
    solutions = mean_field_solution(S[end])  
    
    A = [zeros(ComplexF64, (num_qubits, num_qubits)) for _ in 1:num_layers]
    B = [zeros(ComplexF64, (num_qubits, num_qubits)) for _ in 1:num_layers]
    τ_3 = diagm(vcat(ones(num_qubits), -ones(num_qubits)))

    M = 1.0I(2num_qubits)
    lyapunov_exponent = [zeros(2num_qubits) for _ in 1:num_layers]

    # helper function to construct A and B    
    n_ij_pm = (k, idx, pm) -> solutions[idx] * S[k][idx][1] + pm * 1im * S[k][idx][2]

    for k in 1:size(γ)[1]
        for i in 1:num_qubits
            # we exclude a factor of 2 here because we symmetrize below
            # signs in front of β, γ are reversed relative to the original paper
            A[k][i, i] = -β[k] * S[k][i][1] / (1 + solutions[i] * S[k][i][3]) - γ[k] * solutions[i] * magnetization(S[k], local_fields, couplings)[i]
            for j in i + 1:num_qubits
                A[k][i, j] = γ[k] * couplings[i, j] * n_ij_pm(k, i, 1) * n_ij_pm(k, j, -1)
                B[k][i, j] = γ[k] * couplings[i, j] * n_ij_pm(k, i, 1) * n_ij_pm(k, j,  1)        
            end
        end

        # symmetrize
        A[k] += transpose(conj.(A[k]))
        B[k] += transpose(B[k])

        L = τ_3 * [A[k]                   B[k];
                   transpose(conj.(B[k])) conj.(A[k])]
        
        omega_eig, omega_eigvec = eigen(L)
        
        M = omega_eigvec * diagm(exp.(-1im .* τ .* omega_eig)) * inv(omega_eigvec) * M
        lyapunov_exponential_eig = eigvals(M * transpose(conj.(M)))
        lyapunov_exponent_eig = log.((1.0 + 0.0im) * lyapunov_exponential_eig) ./ 2 .|> real
        lyapunov_exponent_eig = sort(lyapunov_exponent_eig)
        lyapunov_exponent[k] = lyapunov_exponent_eig        
    end

    lyapunov_exponent
end