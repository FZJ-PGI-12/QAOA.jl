"""
    magnetization(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})

Returns the magnetization vector belonging to the input `S`.

### Notes
- The magnetization vector is defined as 

    ``m_i(t) = h_i + \\sum_{j=1}^N J_{ij} n_j^z(t)``.

"""
function magnetization(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})
    h + [sum([J[i, j] * S[j][3] for j in 1:size(S)[1]]) for i in 1:size(S)[1]]
end

"""
    V_P(alpha::Real)

Returns the rotation matrix resulting from the problem Hamiltonian.    
"""
function V_P(alpha::Real) 
    [[cos(alpha), -sin(alpha), 0] [sin(alpha),  cos(alpha), 0] [0,           0,          1]]
end

"""
    V_D(alpha::Real) 

Returns the rotation matrix resulting from the driver Hamiltonian.    
"""
function V_D(alpha::Real) 
    [[1,          0,           0] [0, cos(alpha), -sin(alpha)] [0, sin(alpha),  cos(alpha)]]
end

"""
    evolve(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})

### Input
- `S::Vector{<:Vector{<:Real}}`: The initial vector of all spin vectors.
- `h::Vector{<:Real}`: The vector of local magnetic fields of the problem Hamiltonian.
- `J::Matrix{<:Real}`: The coupling matrix  of the problem Hamiltonian.
- `β::Vector{<:Real}`: The vector of QAOA driver parameters.
- `γ::Vector{<:Real}`: The vector of QAOA problem parameters.

### Output
- The vector of all spin vectors after a full mean-field AOA evolution.

### Notes
- This is the first dispatch of `evolve`, which only returns the final vector of spin vectors.
- The vector of spin vectors is properly initialized as

    `S = [[1., 0., 0.] for _ in 1:num_qubits]`.
"""
function evolve!(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
    @assert size(β)[1] == size(γ)[1] "Invalid QAOA parameters β and γ!"

    for k in 1:size(β)[1]
        # signs in front of β, γ are reversed relative to the original paper
        v_P = V_P.(-2γ[k] * magnetization(S, h, J))
        v_D = V_D.(-2β[k] * ones(size(S)[1]))
        S = [v_D[i] * v_P[i] * S[i] for i in 1:size(S)[1]]
    end    

    S
end

"""
    evolve(S::Vector{<:Vector{<:Vector{<:Real}}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})

### Input
- `S::Vector{<:Vector{<:Vector{<:Real}}}`: An empty history of the vector of all spin vectors.
- `h::Vector{<:Real}`: The vector of local magnetic fields of the problem Hamiltonian.
- `J::Matrix{<:Real}`: The coupling matrix  of the problem Hamiltonian.
- `β::Vector{<:Real}`: The vector of QAOA driver parameters.
- `γ::Vector{<:Real}`: The vector of QAOA problem parameters.

### Output
- The full history of the vector of all spin vectors after a full mean-field AOA evolution.

### Notes
- This is the second dispatch of `evolve`, which returns the full history of the vector of spin vectors.    
- For a schedule of size `num_layers = size(β)[1]`, `S` can be initialized as 

    `S = [[[1., 0., 0.] for _ in 1:num_qubits] for _ in 1:num_layers+1]`.
"""
function evolve!(S::Vector{<:Vector{<:Vector{<:Real}}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
    @assert size(β)[1] == size(γ)[1] "Invalid QAOA parameters β and γ!"

    for k in 1:size(β)[1]
        # signs in front of β, γ are reversed relative to the original paper
        v_P = V_P.(-2γ[k] * magnetization(S[k], h, J))
        v_D = V_D.(-2β[k] * ones(size(h)[1]))
        S[k+1] = [v_D[i] * v_P[i] * S[k][i] for i in 1:size(h)[1]]
    end    

    S
end

"""
    expectation(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})

### Input    
- `S::Vector{<:Vector{<:Real}}`: A vector of all spin vectors.
- `h::Vector{<:Real}`: A vector of local magnetic fields.
- `J::Matrix{<:Real}`: A coupling matrx.

### Output
- The energy expectation value corresponding to the supplied spin configuration.

### Notes
- In the mean-field approximation, the energy expectation value is defined as

    ``
    \\langle E \\rangle = - \\sum_{i=1}^N \\bigg[ h_i + \\sum_{j>i} J_{ij}  n_j^z(p) \\bigg] n_i^z(p).
    ``
"""
function expectation(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})
    S_z = [S[i][3] for i in 1:size(S)[1]]
    # sign is reversed relative to the original paper
    ((transpose(h) .+ 0.5 .* transpose(S_z) * J[1:size(S)[1], 1:size(S)[1]]) * S_z)[1]
end

"""
    mean_field_solution(problem::Problem, β::Vector{<:Real}, γ::Vector{<:Real})

### Input
- `problem::Problem`: A `Problem` instance defining the optimization problem.
- `β::Vector{<:Real}`: The vector of QAOA driver parameters.
- `γ::Vector{<:Real}`: The vector of QAOA problem parameters.

### Output
- The solution bitstring computed for a given problem and schedule parameters.

### Notes
- This is the first dispatch of `mean_field_solution`, which computes the sought-for solution bitstring for a given problem and schedule parameters.
- The solution bitstring ``\\boldsymbol{\\sigma}_*`` is defined as follows in terms of the ``z`` components of the final spin vectors:

    ``
    \\boldsymbol{\\sigma}_* = \\left(\\mathrm{sign}(n_1^z), ..., \\mathrm{sign}(n_N^z) \\right).
    ``
"""
function mean_field_solution(problem::Problem, β::Vector{<:Real}, γ::Vector{<:Real})
    @unpack_Problem problem
    
    @assert size(β)[1] == size(γ)[1] "Invalid QAOA parameters β and γ!"

    # evolution
    S = [[1., 0., 0.] for _ in 1:num_qubits]
    S = QAOA.evolve(S, local_fields, couplings, β, γ)

    # solution (rounded S_z values)
    sign.([S[i][3] for i in 1:size(S)[1]]) 
end

"""
    mean_field_solution(S::Vector{<:Vector{<:Real}})

### Input
- `problem::Problem`: A `Problem` instance defining the optimization problem.
- `β::Vector{<:Real}`: The vector of QAOA driver parameters.
- `γ::Vector{<:Real}`: The vector of QAOA problem parameters.

### Output
- The solution bitstring computed from a given vector of spin vectors.

### Notes
- This is the second dispatch of `mean_field_solution`, which computes the solution bitstring from a given vector of spin vectors.
- The solution bitstring ``\\boldsymbol{\\sigma}_*`` is defined as follows in terms of the ``z`` components of the spin vectors:

    ``
    \\boldsymbol{\\sigma}_* = \\left(\\mathrm{sign}(n_1^z), ..., \\mathrm{sign}(n_N^z) \\right).
    ``
"""
function mean_field_solution(S::Vector{<:Vector{<:Real}})
    # solution (rounded S_z values)
    sign.([S[i][3] for i in 1:size(S)[1]])    
end