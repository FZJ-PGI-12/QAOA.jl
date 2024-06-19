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
    magnetization(S::Matrix{<:Real}, h::Vector{<:Real}, J::Matrix{<:Real})

Second dispatch for the magnetization vector belonging to the matrix input `S`.

### Notes
- The magnetization vector is defined as 

    ``m_i(t) = h_i + \\sum_{j=1}^N J_{ij} n_j^z(t)``.

"""
function magnetization(S::Matrix{<:Real}, h::Vector{<:Real}, J::Matrix{<:Real})
    h + [sum([J[i, j] * S[3, j] for j in 1:size(S)[2]]) for i in 1:size(S)[2]]
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
    evolve!(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})

### Input
- `S::Vector{<:Vector{<:Real}}`: The initial vector of all spin vectors.
- `h::Vector{<:Real}`: The vector of local magnetic fields of the problem Hamiltonian.
- `J::Matrix{<:Real}`: The coupling matrix  of the problem Hamiltonian.
- `β::Vector{<:Real}`: The vector of QAOA driver parameters.
- `γ::Vector{<:Real}`: The vector of QAOA problem parameters.

### Output
- The input `S` is now the vector of all spin vectors after a full mean-field AOA evolution.

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
        S .= [v_D[i] * v_P[i] * S[i] for i in 1:size(S)[1]]
    end    
end

"""
    evolve!(S::Vector{<:Vector{<:Vector{<:Real}}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})

### Input
- `S::Vector{<:Vector{<:Vector{<:Real}}}`: An empty history of the vector of all spin vectors.
- `h::Vector{<:Real}`: The vector of local magnetic fields of the problem Hamiltonian.
- `J::Matrix{<:Real}`: The coupling matrix  of the problem Hamiltonian.
- `β::Vector{<:Real}`: The vector of QAOA driver parameters.
- `γ::Vector{<:Real}`: The vector of QAOA problem parameters.

### Output
- The input `S` is now the full history of the vector of all spin vectors after a full mean-field AOA evolution.

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
end


"""
    evolve(h::Vector{<:Real}, J::Matrix{<:Real}, T_final::Float64, schedule::Function; rtol=1e-4, atol=1e-6)

Evolves the mean-field equations of motion for a given system.

# Input
- `h::Vector{<:Real}`: External magnetic field vector.
- `J::Matrix{<:Real}`: Interaction matrix.
- `T_final::Float64`: Final time of evolution.
- `schedule::Function`: Scheduling function for the evolution.
- `rtol`: Relative tolerance for the ODE solver (default: `1e-4`).
- `atol`: Absolute tolerance for the ODE solver (default: `1e-6`).

# Output
- `sol`: Solution object from the ODE solver containing the time evolution of the system.

# Notes
- This is the third dispatch of `evolve`, which directly solves the full mean-field equations of motion for a system described by an external magnetic field `h` and an interaction matrix `J` over a time interval from `0.0` to `T_final`. The evolution is controlled by a scheduling function `schedule(t)` which interpolates between different dynamical regimes.
- The initial state `S₀` is assumed to be the vector `[1.0, 0.0, 0.0]` for each spin.
- The function uses the `Tsit5()` solver from the `DifferentialEquations.jl` package to solve the ODE.

# Example
```julia
using DifferentialEquations

h = [0.5, -0.5, 0.3]
J = [0.0 0.1 0.2; 0.1 0.0 0.3; 0.2 0.3 0.0]
T_final = 10.0
schedule(t) = t / T_final

sol = evolve_mean_field(h, J, T_final, schedule)
```
"""
function evolve(h::Vector{<:Real}, J::Matrix{<:Real}, T_final::Float64, schedule::Function; rtol=1e-4, atol=1e-6)

    function mf_eom(dS, S, _, t)
        magnetization = h + [sum([J[i, j] * S[3, j] for j in 1:size(S)[2]]) for i in 1:size(S)[2]]
        dS .= reduce(hcat, [[-2 * schedule(t) * magnetization[i] * S[2, i], 
                             -2 * (1 - schedule(t)) * S[3, i] + 2 * schedule(t) * magnetization[i] * S[1, i],
                              2 * (1 - schedule(t)) * S[2, i]] for i in 1:size(S)[2]])
    end

    S₀ = reduce(hcat, [[1., 0., 0.] for _ in 1:size(h)[1]])
    prob = ODEProblem(mf_eom, S₀, (0.0, T_final))
    sol = solve(prob, Tsit5(), reltol=rtol, abstol=atol)
    sol
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
    evolve!(S, local_fields, couplings, β, γ)

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