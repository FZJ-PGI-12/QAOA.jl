function magnetization(S::Vector{<:Vector{<:Real}}, J::Matrix{<:Real})
    J[size(S)[1]+1, 1:size(S)[1]] + [sum([J[i, j] * S[j][3] for j in 1:size(S)[1]]) for i in 1:size(S)[1]]
end

function V_P(alpha::Real) 
    [[cos(alpha), -sin(alpha), 0] [sin(alpha),  cos(alpha), 0] [0,           0,          1]]
end

function V_D(alpha::Real) 
    [[1,          0,           0] [0, cos(alpha), -sin(alpha)] [0, sin(alpha),  cos(alpha)]]
end


function evolve(S::Vector{<:Vector{<:Real}}, J::Matrix{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real})
    @assert size(β)[1] == size(γ)[1]

    for k in 1:size(β)[1]
        v_P = V_P.(2γ[k] * magnetization(S, J))
        v_D = V_D.(2β[k] * ones(size(S)[1]))
        S = [v_D[i] * v_P[i] * S[i] for i in 1:size(S)[1]]
    end    

    S
end