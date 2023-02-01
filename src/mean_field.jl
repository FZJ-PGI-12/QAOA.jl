"""
    magnetization(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})
"""
function magnetization(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})
    h + [sum([J[i, j] * S[j][3] for j in 1:size(S)[1]]) for i in 1:size(S)[1]]
end

"""
    V_P(alpha::Real)
"""
function V_P(alpha::Real) 
    [[cos(alpha), -sin(alpha), 0] [sin(alpha),  cos(alpha), 0] [0,           0,          1]]
end

"""
    V_D(alpha::Real) 
"""
function V_D(alpha::Real) 
    [[1,          0,           0] [0, cos(alpha), -sin(alpha)] [0, sin(alpha),  cos(alpha)]]
end

"""
    evolve(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
"""
function evolve(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
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
"""
function evolve(S::Vector{<:Vector{<:Vector{<:Real}}}, h::Vector{<:Real}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
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
"""
function expectation(S::Vector{<:Vector{<:Real}}, h::Vector{<:Real}, J::Matrix{<:Real})
    S_z = [S[i][3] for i in 1:size(S)[1]]
    ((transpose(h) .+ 0.5 .* transpose(S_z) * J[1:size(S)[1], 1:size(S)[1]]) * S_z)[1]
end

"""
    mean_field_solution(problem::Problem, β::Vector{<:Real}, γ::Vector{<:Real})
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
"""
function mean_field_solution(S::Vector{<:Vector{<:Real}})
    # solution (rounded S_z values)
    sign.([S[i][3] for i in 1:size(S)[1]])    
end