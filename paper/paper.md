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
---

# Summary

Quantum algorithms are an area of intensive research thanks to their potential of speeding up certain specific tasks exponentially. However, for the time being, high error rates on the existing hardware realizations preclude the application of many algorithms that are based on the assumption of fault-tolerant quantum computation. On such _noisy intermediate-scale quantum_ (NISQ) devices [@Preskill2018], the exploration of the potential of _heuristic_ quantum algorithms has attracted much interest. A leading candidate for solving combinatorial optimization problems is the so-called _Quantum Approximate Optimization Algorithm_ (QAOA) [@Farhi:2014].

`QAOA.jl` is a `Julia` package [@bezanson2017julia] package that implements the _mean-field Approximate Optimization Algorithm_ (mean-field AOA) [@mf_aoa] - a quantum-inspired classical algorithm derived from the QAOA via the mean-field approximation. This novel algorithm is useful in assisting the search for quantum advantage by providing a tool to discriminate (combinatorial) optimization problems that can be solved classically from those that cannot.  Note that `QAOA.jl` has already been used during the research leading to [@mf_aoa].

Additionally, `QAOA.jl`  also implements the QAOA efficiently to support the extensive classical simulations typically required in research on the topic. The corresponding parameterized circuits are based on `Yao.jl` [@YaoFramework2019], [@Yao] and `Zygote.jl` [@ZygoteFramework], [@Zygote], making it both fast and automatically differentiable, thus enabling gradient-based optimization. A number of common optimization problems such as MaxCut, the minimum vertex-cover problem, the Sherrington-Kirkpatrick model, and the partition problem are pre-implemented to facilitate scientific benchmarking.

# Statement of need

The demonstration of quantum advantage for a real-world problem is yet outstanding. Identifying such a problem and performing the actual demonstration on existing hardware will not be possible without intensive (classical) simulations. `QAOA.jl` facilitates this exploration by offering a classical baseline through the mean-field AOA, complemented by a fast and versatile implementation of the QAOA. As shown in our benchmarks, QAOA simulations performed with `QAOA.jl` are significantly faster than those of `PennyLane` [@PennyLane], one of its main competitors in automatically differentiable QAOA implementations. While Tensorflow Quantum [@tfq] supports automatic differentiation, there exists, to the authors's knowledge, no dedicated implementation of the QAOA. The class `QAOA` offered by Qiskit [@Qiskit] must be _provided_ with a precomputed gradient operator, i.e. it does not feature automatic differentiation out of the box.


# Acknowledgements

The authors acknowledge partial support from the German Federal Ministry of Education and Research, under the funding program "Quantum technologies - from basic research to the market", Contract Numbers 13N15688 (DAQC), 13N15584 (Q(AI)2) and from the German Federal Ministry of Economics and Climate Protection under contract number, 01MQ22001B (Quasim).


# References