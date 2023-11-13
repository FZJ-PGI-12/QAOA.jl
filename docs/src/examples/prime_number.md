# [Prime-Number QUBOs with the Mean-Field AOA] (@id QANN)

!!! note
     The paper [Factoring integers with sublinear resources on a superconducting quantum processor](https://arxiv.org/pdf/2212.12372.pdf) derives a QUBO problem equivalent to factorizing the 48-bit integer `261980999226229`. Here, we show that the ground state of this QUBO Hamiltonian can also be obtained with the mean-field AOA. Have a look into the corresponding [notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/prime_number.ipynb).

The Hamiltonian taken from the above paper reads
```math
\begin{aligned}
H_{c 10} & =\left(708 I+22 \sigma_z^1 \sigma_z^2+16 \sigma_z^1 \sigma_z^3+8 \sigma_z^1 \sigma_z^4-14 \sigma_z^1 \sigma_z^5+8 \sigma_z^1 \sigma_z^6+4 \sigma_z^1 \sigma_z^7-8 \sigma_z^1 \sigma_z^8-10 \sigma_z^1 \sigma_z^9-22 \sigma_z^1 \sigma_z^{10}-46 \sigma_z^1-14 \sigma_z^2 \sigma_z^3\right. \\
& +20 \sigma_z^2 \sigma_z^4+14 \sigma_z^2 \sigma_z^5-12 \sigma_z^2 \sigma_z^6+2 \sigma_z^2 \sigma_z^7-24 \sigma_z^2 \sigma_z^8-28 \sigma_z^2 \sigma_z^9+2 \sigma_z^2 \sigma_z^{10}-16 \sigma_z^2-18 \sigma_z^3 \sigma_z^4+10 \sigma_z^3 \sigma_z^5+36 \sigma_z^3 \sigma_z^6+12 \sigma_z^3 \sigma_z^7 \\
& +16 \sigma_z^3 \sigma_z^8+6 \sigma_z^3 \sigma_z^9-30 \sigma_z^3 \sigma_z^{10}-78 \sigma_z^3+28 \sigma_z^4 \sigma_z^5-26 \sigma_z^4 \sigma_z^6+10 \sigma_z^4 \sigma_z^7+10 \sigma_z^4 \sigma_z^8+16 \sigma_z^4 \sigma_z^9-4 \sigma_z^4 \sigma_z^{10}-72 \sigma_z^4+10 \sigma_z^5 \sigma_z^6 \\
& +24 \sigma_z^5 \sigma_z^7+20 \sigma_z^5 \sigma_z^8+12 \sigma_z^5 \sigma_z^9-8 \sigma_z^5 \sigma_z^{10}-116 \sigma_z^5-8 \sigma_z^6 \sigma_z^7+22 \sigma_z^6 \sigma_z^8-6 \sigma_z^6 \sigma_z^9-36 \sigma_z^6 \sigma_z^{10}-12 \sigma_z^6-16 \sigma_z^7 \sigma_z^8+16 \sigma_z^7 \sigma_z^9 \\
& \left.+20 \sigma_z^7 \sigma_z^{10}-84 \sigma_z^7+34 \sigma_z^8 \sigma_z^9-42 \sigma_z^8 \sigma_z^{10}-36 \sigma_z^8+18 \sigma_z^9 \sigma_z^{10}-74 \sigma_z^9-24 \sigma_z^{10}\right) / 4.
\end{aligned}
```
In terms of `QAOA.jl`, we thus have `local_fields`
```math
h =  - 46 \sigma_z^1 - 16 \sigma_z^2 - 78 \sigma_z^3 - 72 \sigma_z^4 - 116 \sigma_z^5 - 12 \sigma_z^6 - 84 \sigma_z^7 - 36 \sigma_z^8 - 74 \sigma_z^9 - 24 \sigma_z^{10}
```
and `couplings`
```math
\begin{aligned}
J = 4 H_{c 10} - 708 I - h& =  22 \sigma_z^1 \sigma_z^2 + 16 \sigma_z^1 \sigma_z^3 + 8 \sigma_z^1 \sigma_z^4 - 14 \sigma_z^1 \sigma_z^5 + 8 \sigma_z^1 \sigma_z^6 + 4 \sigma_z^1 \sigma_z^7 - 8 \sigma_z^1 \sigma_z^8 - 10 \sigma_z^1 \sigma_z^9 - 22 \sigma_z^1 \sigma_z^{10}  \\
&  - 14 \sigma_z^2 \sigma_z^3 + 20 \sigma_z^2 \sigma_z^4 + 14 \sigma_z^2 \sigma_z^5 - 12 \sigma_z^2 \sigma_z^6 + 2 \sigma_z^2 \sigma_z^7 - 24 \sigma_z^2 \sigma_z^8 - 28 \sigma_z^2 \sigma_z^9 + 2 \sigma_z^2 \sigma_z^{10} \\
& - 18 \sigma_z^3 \sigma_z^4 + 10 \sigma_z^3 \sigma_z^5 + 36 \sigma_z^3 \sigma_z^6 + 12 \sigma_z^3 \sigma_z^7   + 16 \sigma_z^3 \sigma_z^8 + 6 \sigma_z^3 \sigma_z^9 - 30 \sigma_z^3 \sigma_z^{10} \\
& + 28 \sigma_z^4 \sigma_z^5 - 26 \sigma_z^4 \sigma_z^6 + 10 \sigma_z^4 \sigma_z^7 + 10 \sigma_z^4 \sigma_z^8 + 16 \sigma_z^4 \sigma_z^9 - 4 \sigma_z^4 \sigma_z^{10} \\
& + 10 \sigma_z^5 \sigma_z^6  + 24 \sigma_z^5 \sigma_z^7 + 20 \sigma_z^5 \sigma_z^8 + 12 \sigma_z^5 \sigma_z^9 - 8 \sigma_z^5 \sigma_z^{10}\\
& - 8 \sigma_z^6 \sigma_z^7 + 22 \sigma_z^6 \sigma_z^8 - 6 \sigma_z^6 \sigma_z^9 - 36 \sigma_z^6 \sigma_z^{10} \\
& - 16 \sigma_z^7 \sigma_z^8 + 16 \sigma_z^7 \sigma_z^9 + 20 \sigma_z^7 \sigma_z^{10} \\
& + 34 \sigma_z^8 \sigma_z^9 - 42 \sigma_z^8 \sigma_z^{10}  \\
& + 18 \sigma_z^9 \sigma_z^{10}.
\end{aligned}
```
Accordingly, we implement
```julia
using QAOA, Printf

h = [-46., -16, -78, -72, -116, -12, -84, -36, -74, -24]

J = [[0., 22, 16, 8, -14, 8, 4, -8, -10, -22],
     [0, 0, -14, 20, 14, -12, 2, -24, -28, 2],
     [0, 0, 0, -18, 10, 36, 12, 16, 6, -30],
     [0, 0, 0, 0, 28, -26, 10, 10, 16, -4],
     [0, 0, 0, 0, 0, 10, 24, 20, 12, -8],
     [0, 0, 0, 0, 0, 0, -8, 22, -6, -36],
     [0, 0, 0, 0, 0, 0, 0, -16, 16, 20],
     [0, 0, 0, 0, 0, 0, 0, 0, 34, -42],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 18],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

J = reduce(hcat, J)
J = J + transpose(J)

N = size(h)[1]
```
The true solution of the QUBO Hamiltonian is
```julia
true_sol = "0100010010"
```
Now, we set up our standard mean-field schedule as
```julia
# slow evolution (works)
# p = 100000
# τ = 0.01

# relatively fast evolution (also works)
p = 5000
τ = 0.03 

# schedule
γ = τ .* ((1:p) .- 1/2) ./ p |> collect
β = τ .* (1 .- (1:p) ./ p) |> collect
β[p] = τ / (4 * p)
```
and call the mean-field AOA:
```julia
# we need to take minus the fields by our convention
mf_problem = Problem(p, -h, -J)
S = [[1., 0., 0.] for _ in 1:N]

S = evolve!(S, mf_problem.local_fields, mf_problem.couplings, β, γ)
```
As you can verify with this [notebook](https://github.com/FZJ-PGI-12/QAOA.jl/blob/master/notebooks/prime_number.ipynb), the solution returned by our algorithm agrees with the true solution:
```julia
sol = mean_field_solution(S)
prod(map(x -> @sprintf("%i", x), (1 .- sol) ./ 2)) |> println
```
outputs `0100010010`.