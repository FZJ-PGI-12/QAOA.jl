# [Quantum Annealing with `QAOA.jl`] (@id QANN)

!!! tip
     Have a look into our corresponding [SK](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/sherrington_kirkpatrick.ipynb) and [MaxCut](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/max_cut.ipynb) notebooks.

Simulating quantum annealing with `QAOA.jl` is very simple. For the most basic case of a *linear* annealing schedule, all we need is to define a schedule function for a certain final time `T_anneal` and a `Problem` instance:
```julia
using QAOA

T_anneal = 8.
p = 256
linear_schedule(t) = t / T_anneal
annealing_problem = Problem(p, zeros(N), J)
```
Note that we assume `J` has been defined and that no `local_fields` are present (i.e. `local_fields = zeros(N)`). Then we can call `QAOA.jl`'s `anneal` method to obtain the final probabilities of each bitstring:
```julia
probs = anneal(annealing_problem, linear_schedule, T_anneal)
```
Two specific examples of this are implemented in the above-mentioned notebooks.