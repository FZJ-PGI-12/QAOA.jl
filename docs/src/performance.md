# [Performance](@id Performance)

`QAOA.jl` does not build a quantum circuit. It applies each layer of the
QAOA to the state vector directly, with loops written for the two parts of a
layer. This page explains these loops and gives timings, so that you can
estimate how large a problem your computer can handle.

## Timings

The benchmark of Fig. 4 in [Golden et al., arXiv:2312.06451](https://arxiv.org/abs/2312.06451)
times one expectation value of MaxCut on a random graph `G(n, 0.5)` with one
layer. On an Apple M4 with Julia 1.12 and one thread:

| qubits | QAOA.jl 1.4 | JuliQAOA 0.1 | QAOA.jl 2.0 | QAOA.jl 2.0, halved by symmetry |
|---|---|---|---|---|
| 6  | 149 µs | 1.3 µs  | 0.32 µs | 0.22 µs |
| 14 | 3.3 ms | 0.39 ms | 0.11 ms | 0.057 ms |
| 20 | 464 ms | 30 ms   | 9.6 ms  | 4.8 ms |
| 24 |        |         | 183 ms  | 92 ms |

All three packages agree on the expectation values to 13 digits. QAOA.jl 1.4
allocated 3.6 GB per evaluation at 20 qubits; QAOA.jl 2.0 allocates nothing.
With four threads, 20 qubits take 1.7 ms and 24 qubits 35 ms, halved by
symmetry. The expectation value and its derivatives with respect to all
angles cost about three times one expectation value: for 14 qubits and 10
layers, 3.5 ms, against 14 ms for JuliQAOA and 84 ms for QAOA.jl 1.4.

## The two parts of a layer

The phase separator multiplies every amplitude by a phase that depends only
on the objective of its assignment. The [`Workspace`](@ref) computes these
objectives once, for all `2^n` assignments. For an Ising or QUBO problem,
adding a variable doubles the table of objectives, so the whole table costs
`O(2^n)` operations; for 20 qubits, less than a millisecond. When all weights are integers,
or multiples of `1/16`, the objective takes at most 65535 values, and the
workspace stores a two-byte index into a table of these values instead of an
eight-byte number. The phase separator then computes one complex exponential
per distinct value instead of one per assignment.

The mixer turns each qubit separately. The amplitudes are read and written
once per pass over the state vector, and for a large state, memory traffic
sets the time. The workspace therefore turns the eleven lowest qubits block by
block, inside 32 KiB of the state at a time, together with the phase
separator. The remaining qubits are turned two per pass.

## Halving by symmetry

Many objectives do not change when all variables flip at once: MaxCut, the
Sherrington–Kirkpatrick model and number partitioning, for example. Then the
two amplitudes of an assignment and of its complement stay equal throughout
the QAOA, and the workspace stores only one of them. This halves memory and
time and changes no result. A workspace detects the symmetry; `symmetry =
false` turns it off. It needs the X mixer and the default initial state.

## Gradients

[`expectation_and_gradient`](@ref) computes the derivatives of the expectation
value with respect to all `2p` angles by the adjoint method: after the forward
run, it goes back through the layers once, undoing them on the state and on a
second vector. This costs about two more runs, for any number of layers, and
one more state vector of memory.

## Threads

The passes over the state vector split over Julia's threads when the state
has at least `2^15` amplitudes. Start Julia with several threads, for example
`julia --threads=auto`. Smaller states run on one thread: for them, starting
threads costs more than it saves. To use all cores for many small problems,
run one problem per thread instead.

## Memory and the largest problems

Every stored assignment takes 16 bytes for its amplitude and 8 bytes for its
objective, or 2 bytes with integer weights. A gradient needs another 16 bytes
per assignment. Halving by symmetry halves all of this.

| qubits | state | objective | with symmetry |
|---|---|---|---|
| 24 | 0.27 GB | 0.13 GB | half |
| 28 | 4.3 GB  | 2.1 GB  | half |
| 30 | 17 GB   | 8.6 GB  | half |
| 32 | 69 GB   | 34 GB   | half |

A computer with 24 GB of memory thus handles about 29 qubits, or 30 halved by
symmetry. One layer with 28 qubits takes 1.4 s on four cores of an Apple M4,
or 0.7 s halved by symmetry.
