# [Number partitioning](@id Partition)

!!! note
    A [Jupyter notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/partition_problem.ipynb)
    with this example is available in the [notebooks folder](https://github.com/FZJ-PGI-12/QAOA.jl/tree/master/notebooks).

The partition problem splits a list of numbers into two groups whose sums are
as close as possible. Spin `σᵢ = ±1` says which group the number `aᵢ` joins,
so the difference of the two sums is `Σᵢ aᵢ σᵢ`. [`partition_problem`](@ref)
minimizes its square,

```math
C = \Big(\sum_i a_i \sigma_i\Big)^2 = \sum_i a_i^2 + 2 \sum_{i<j} a_i a_j \sigma_i \sigma_j.
```

The seven numbers below add up to 14, so a perfect split has two groups of
sum 7:

```@example partition
using QAOA, Random
a = [3, 1, 1, 2, 2, 1, 4]
result = solve(partition_problem(a); p = 40, rng = Xoshiro(1))
```

The objective 0 means a perfect split. The spins give the two groups:

```@example partition
a[result.σ .== 1], a[result.σ .== -1]
```
