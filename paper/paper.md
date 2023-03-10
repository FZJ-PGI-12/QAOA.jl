---
title: 'QAOA.jl: Toolkit for the Quantum and Mean-Field Approximate Optimization Algorithms'
tags:
  - Julia
  - quantum algorithms
  - automatic differentiation
  - optimization
authors:
  - name: Tim Bode
    orcid: 0000-0001-8280-3891
    corresponding: true
    affiliation: 1
  - name: Dmitry Bagrets
    affiliation: "1, 2"
  - name: Aditi Misra-Spieldenner
    affiliation: 3
  - name: Tobias Stollenwerk
    affiliation: 1
  - name: Frank K. Wilhelm
    affiliation: "1, 3"    
affiliations:
 - name: Institute for Quantum Computing Analytics (PGI-12), Forschungszentrum Jülich, 52425 Jülich, Germany
   index: 1
 - name: Institute for Theoretical Physics, University of Cologne, 50937 Cologne, Germany
   index: 2  
 - name: Theoretical Physics, Saarland University, 66123 Saarbrücken, Germany
   index: 3
date: 19 January 2023
bibliography: paper.bib


# Summary

Quantum algorithms are an area of intensive research thanks to their potential of speeding up certain specific tasks exponentially. However, for the time being, high error rates on the existing hardware realizations preclude the application of many algorithms that are based on the assumption of fault-tolerant quantum computation. On such _noisy intermediate-scale quantum_ (NISQ) devices [@Preskill2018], the exploration of the potential of _heuristic_ quantum algorithms has attracted much interest. A leading candidate for solving combinatorial optimization problems is the so-called _Quantum Approximate Optimization Algorithm_ (QAOA) [@Farhi:2014]. `QAOA.jl` is a `Julia` package [@bezanson2017julia] that implements the QAOA to enable the efficient classical simulation typically required in research on the topic. It is based on `Yao.jl` [@YaoFramework2019], [@Yao] and `Zygote.jl` [@ZygoteFramework], [@Zygote], making it both fast and automatically differentiable, thus enabling gradient-based optimization. A number of common optimization problems such as MaxCut, the minimum vertex-cover problem, the Sherrington-Kirkpatrick model, and the partition problem are pre-implemented to facilitate scientific benchmarking. 

Additionally, `QAOA.jl` is the first package to implement the _mean-field Approximate Optimization Algorithm_ (mean-field AOA) [@mf_aoa], which is a quantum-inspired classical algorithm derived from the QAOA via the mean-field approximation. Note that `QAOA.jl` has already been used during the research leading to [@mf_aoa]. This novel algorithm can be useful in assisting the search for quantum advantage since it provides a tool to discriminate (combinatorial) optimazation problems that be solved classically from those that cannot.


# Statement of need

The demonstration of quantum advantage for a real-world problem is yet outstanding. Identifying such a problem and performing the actual demonstration on existing hardware will not be possible without intensive (classical) simulations. This makes a fast and versatile implementation of the QAOA rather desirable. As shown in \autoref{fig:benchmarks}, `QAOA.jl` is significantly faster than `PennyLane` [@PennyLane], one of its main competitors in automatically differentiable QAOA implementations. While Tensorflow Quantum [@tfq] supports automatic differentiation, there exists, to the author's knowledge, no dedicated implementation of the QAOA. The class `QAOA` offered by Qiskit [@Qiskit] must be _provided_ with a precomputed gradient operator, i.e. it does not feature automatic differentiation out of the box.

As already mentioned, `QAOA.jl` is also the first package to implement the mean-field AOA, which is thus made available to researchers working on the topic.


![Comparison of run times between `PennyLane` [@PennyLane] and `QAOA.jl` on an Apple M1 processor. The benchmarks $\Delta t$ are retrieved by performing 128 steps with the respective gradient optimizer on the same instance of size $N$ of the minimum vertex-cover problem.\label{fig:benchmarks}](benchmarks.pdf)


# Mathematics 

## QAOA

The cost function of the QAOA for a general quadratic optimization problem is typically defined as
$$
  \hat C = \sum_{i=1}^N \bigg[ h_i  + \sum_{j>i} J_{ij}  \hat{Z}_j \bigg] \hat{Z}_i,
$$
where the $h_i$, $J_{ij}$ are real numbers encoding the problem in question, and $\hat Z_{i, j}$ denote Pauli matrices. Similarly, the conventional _mixer_ or _driver_ of the QAOA is given by
$$
  \hat D = \sum_{i=1}^N \hat X_i,
$$
where the $\hat X_i$ are again Pauli matrices. We also introduce the initial quantum state
$$
  |\psi_0\rangle = |+\rangle_1 \otimes \cdots \otimes |+\rangle_N.
$$
Note that this is the maximum-energy eigenstate of the driver $\hat D$ since $\langle \psi_0 | \hat D | \psi_0 \rangle = N$. With these prerequisites, the variational quantum state of the QAOA becomes
$$
 |\psi(\boldsymbol{\beta}, \boldsymbol{\gamma})\rangle = \exp{\left(-\mathrm{i}\beta_p\hat D\right)}\exp{\left(-\mathrm{i}\gamma_p\hat C\right)}\cdots \exp{\left(-\mathrm{i}\beta_1\hat D\right)}\exp{\left(-\mathrm{i}\gamma_1\hat C\right)}|\psi_0\rangle.
$$
The goal is then to _maximize_ the expectation value
$$
  E_p(\boldsymbol{\beta}, \boldsymbol{\gamma}) = \langle\psi(\boldsymbol{\beta}, \boldsymbol{\gamma})| \hat C |\psi(\boldsymbol{\beta}, \boldsymbol{\gamma})\rangle
$$
over the variational parameters $\boldsymbol{\beta}, \boldsymbol{\gamma}$.
Note that `QAOA.jl` furthermore supports others drivers, e.g. 
$$
\hat D = \sum_{(i, j)\in\mathcal{E}}  \left(\hat X_i \hat X_j + \hat Y_i \hat Y_j\right),
$$
where $\mathcal{E}$ is the set of connections or _edges_ for which the coupling matrix $J_{ij}$ is non-zero.

## Mean-Field AOA

In close analogy to the QAOA, the mean-field Hamiltonian reads
$$
    H(t) ={}  \gamma(t)\sum_{i=1}^N \bigg[ h_i + \sum_{j>i} J_{ij}  n_j^z(t) \bigg] n_i^z(t) + \beta(t) \sum_{i=1}^N n_i^x(t).
$$
The mean-field evolution is then given by
$$
    \boldsymbol{n}_i(p) = \prod_{k=1}^p \hat V_i^D(k) \hat V_i^P(k) \boldsymbol{n}_i(0),
$$
where the initial spin vectors are $\boldsymbol{n}_i(0) = (1, 0, 0)^T$ for all $i=1, ..., N$, and the rotation matrices $\hat V_i^{D,\,P}$ are defined as
$$
\hat V_i^D(k) = 
\begin{pmatrix}
1 & 0 & 0 \\
0 & \cos(2 \Delta_i\beta_k) & -\sin(2 \Delta_i \beta_k) \\
0 & \sin (2 \Delta_i \beta_k) & \phantom{-}\cos(2 \Delta_i \beta_k) 
\end{pmatrix}
$$
and
$$
\hat V_i^P(k) = 
\begin{pmatrix}
\cos(2m_i (t_{k-1}) \gamma_k) & -\sin(2m_i (t_{k-1}) \gamma_k) & 0 \\
\sin (2m_i (t_{k-1}) \gamma_k) & \phantom{-}\cos(2m_i (t_{k-1}) \gamma_k) & 0 \\
0 & 0 & 1
\end{pmatrix},
$$
with the magnetization 
$$
m_i(t) = h_i + \sum_{j=1}^N J_{ij} n_j^z(t).
$$


# Acknowledgements

The authors acknowledge partial support from the German Federal Ministry of Education and Research, under the funding program "Quantum technologies - from basic research to the market", Contract Numbers 13N15688 (DAQC), 13N15584 (Q(AI)2) and from the German Federal Ministry of Economics and Climate Protection under contract number, 01MQ22001B (Quasim).


# References