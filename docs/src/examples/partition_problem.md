# [Partition Problem](@id Partition)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/partition_problem.ipynb) related to this example is available in our [examples folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The partition problem (see also [Wikipedia](https://en.wikipedia.org/wiki/Partition_problem)) for a set of numbers ``\mathcal{S} = \{a_1, ..., a_N\}`` consists of finding two subsets ``\mathcal{S}_{1} \cup \mathcal{S}_2 = \mathcal{S}`` such that the difference of the sums over the two subsets ``\mathcal{S}_{1, 2}`` is as small as possible. Spin ``\sigma_i = \pm 1`` says which subset ``a_i`` joins. The cost function in Ising form can then be defined as
```math
\begin{align*}
C(\sigma) = \left(\sum_{i=1}^{N} a_i \sigma_i\right)^2 = \sum_{i<j\leq N} J_{ij} \sigma_i \sigma_j + \sum_{i=1}^{N} a_i^2
\end{align*}
```
with ``J_{ij} = 2a_i a_j``. The goal is then to _minimize_ ``C``.

As an example, we take seven numbers that add up to 14, so a perfect split has two subsets with the sum 7. We can set this model up as follows:
```@example partition
using QAOA, LinearAlgebra, Random

a = [3, 1, 1, 2, 2, 1, 4]
N = length(a)
J = 2 .* (a * transpose(a))
J[diagind(J)] .= 0
nothing # hide
```
We have two options to get the corresponding [`IsingProblem`](@ref) from `QAOA.jl`. We can pass the coupling matrix `J` directly, together with the constant ``\sum_i a_i^2``:
```@example partition
number_partition = IsingProblem(zeros(N), J; offset = sum(abs2, a))
```
or we can use a predefined wrapper function that constructs `J` from `a` and directly returns an [`IsingProblem`](@ref):
```@example partition
number_partition = partition_problem(a)
```
Given `number_partition`, we can then run a quantum annealing with 40 layers and measure the final state with [`solve`](@ref):
```@example partition
result = solve(number_partition; p = 40, rng = Xoshiro(1))
```
The objective 0 means a perfect split. The spins give the two subsets:
```@example partition
a[result.σ .== 1], a[result.σ .== -1]
```
