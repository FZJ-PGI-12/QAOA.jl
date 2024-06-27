"""
    trajectory_distance(sol_t::Vector, n_i::Vector, n_j::Vector)

Calculates the distance between two trajectories over time, taking into account the time intervals between solution points.

# Input
- `sol_t::Vector`: A vector of time points corresponding to the solution trajectory.
- `n_i::Vector`: A vector representing the first trajectory.
- `n_j::Vector`: A vector representing the second trajectory.

# Output
- `Float64`: The distance between the two trajectories, averaged over the total time span.

# Notes
- This function computes the distance between two trajectories `n_i` and `n_j` by first calculating the time intervals (`Δts`) between each consecutive pair of time points in `sol_t`. It then calculates the element-wise absolute difference between `n_i` and `n_j`, multiplies these differences by the corresponding time intervals, and sums up the results. Finally, the sum is divided by the total time span (the last element of `sol_t`) to get the average distance over the time span.
"""
function trajectory_distance(sol_t::Vector, n_i::Vector, n_j::Vector)
    Δts = map(x -> x[2] - x[1], zip(sol_t[1:end-1], sol_t[2:end]))
    sum(Δts .* abs.(n_i .- n_j)) / sol_t[end]
end

"""
    distance_matrix(sol_t::Vector, n_vals::Matrix)

Calculates the distance matrix for a set of trajectories over time.

# Input
- `sol_t::Vector`: A vector of time points corresponding to the solution trajectory.
- `n_vals::Matrix`: A matrix where each row represents the trajectory of a different qubit. The first column is assumed to be the initial conditions and is not included in the distance calculation.

# Output
- `Matrix{Float64}`: A symmetric matrix where the entry at (i, j) represents the distance between the trajectories of qubit `i` and qubit `j`.

# Notes
- This function computes the pairwise distances between the trajectories of multiple qubits. It initializes a distance matrix with zeros and then iterates over pairs of qubits, calculating the distance between their trajectories using the `trajectory_distance` function. The resulting distance matrix is symmetric, with distances mirrored across the diagonal.
"""
function distance_matrix(sol_t::Vector, n_vals::Matrix)
    num_qubits = size(n_vals)[1]
    dist_mat = zeros(num_qubits, num_qubits)
    for idx1 in 1:num_qubits
        for idx2 in idx1+1:num_qubits
            dist_mat[idx1, idx2] = trajectory_distance(sol_t, n_vals[idx1, 2:end], n_vals[idx2, 2:end])
        end
    end
    dist_mat + dist_mat'
end

"""
    distance_clusters(dist_mat::Matrix{Float64}; dist::Float64=0.2)

Clusters qubits based on their pairwise mean-field trajectory distances.

# Input
- `dist_mat::Matrix{Float64}`: A symmetric distance matrix where the entry at (i, j) represents the distance between the trajectories of qubit `i` and qubit `j`.
- `dist::Float64`: The distance threshold below which qubits are considered to be in the same cluster. Default is `0.2`.

# Output
- `Vector{Vector{Int}}`: A list of clusters, where each cluster is represented by a vector of qubit indices.

# Notes
- This function clusters qubits based on their pairwise distances. It iterates through all qubits and groups them into clusters if the distance between them is below the specified threshold. The function returns only clusters that contain more than one qubit.
"""
function distance_clusters(dist_mat, dist=0.2)
    num_qubits = size(dist_mat)[1]
    clusters = []
    clustered_idxs = []
    for idx1 in 1:num_qubits
        cluster = []
        # do not consider index if already part of a cluster
        if isempty(filter(idx -> idx == idx1, clustered_idxs))
            push!(cluster, idx1)
            push!(clustered_idxs, idx1)
        end
        for idx2 in idx1+1:num_qubits
            # do not consider index if already part of a cluster
            if isempty(filter(idx -> idx == idx2, clustered_idxs))
                if 0 < dist_mat[idx1, idx2] < dist
                    push!(cluster, idx2)
                    push!(clustered_idxs, idx2)
                end
            end   
        end
        push!(clusters, cluster) 
    end
    filter!(c -> size(c)[1] > 1, clusters)
end

"""
    filtered_clusters(sol_t::Vector{Float64}, n_vals::Matrix{Float64}; cutoff::Float64=0.5)

Clusters qubits based on their pairwise distances and filters clusters based on integrated trajectory values.

# Input
- `sol_t::Vector{Float64}`: A vector of time points corresponding to the solution trajectory.
- `n_vals::Matrix{Float64}`: A matrix where each row corresponds to the trajectory of a qubit.
- `cutoff::Float64`: The threshold for filtering clusters based on the absolute value of the integrated trajectory. Default is `0.5`.

# Output
- `Vector{Vector{Int}}`: A list of filtered clusters, where each cluster is represented by a vector of qubit indices. Only clusters containing qubits with integrated trajectory values above the cutoff are included.

# Notes
- This function first clusters qubits based on their pairwise distances using the `distance_clusters` function. It then filters the clusters based on the integrated trajectory values of the qubits. A qubit is included in a filtered cluster only if the absolute value of its integrated trajectory is greater than the specified cutoff. The function returns only non-empty clusters.
"""
function filtered_clusters(sol_t::Vector, n_vals::Matrix, dist=0.2, cutoff=0.5)
    clusters = distance_clusters(distance_matrix(sol_t, n_vals), dist)
    
    Δts = map(x -> x[2] - x[1], zip(sol_t[1:end-1], sol_t[2:end]))
    integrated_trajectories = Dict(idx => sum(Δts .* n_vals[idx, 2:end]) ./ sol_t[end] for idx in 1:size(n_vals)[1])    
    
    filtered_clusters = []
    for cluster in clusters
        filtered_cluster = []
        for idx in cluster
            if abs(integrated_trajectories[idx]) > cutoff
                push!(filtered_cluster, idx)
            end
        end
        push!(filtered_clusters, filtered_cluster)
    end
    filter!(c -> size(c)[1] > 1, filtered_clusters)
end