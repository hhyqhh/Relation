# Relation Model

Ths repository has collected some of our research work and code implementations on the **relation model**. The Matlab version implementation of the **relation model** can be found [here](https://github.com/hhyqhh/RelationModel-matlab). 


---


## Implementation
| algorithm | python | matlab |
| --------- | :----: | :----: |
| RCPS      |   ✅    |        |
| RCPS-MOEA |        |   ✅    |
| REMO      |        |   ✅    |
| CREMO     |        |   ✅    |
| DRSO      |   ✅    |        |



### RCPS 

In this work, we propose using relation models for pre-selection operations to accelerate the convergence speed of evolutionary algorithms. ([code](), [paper](https://ieeexplore.ieee.org/document/9060983))


>Hao H, Zhang J, Lu X, et al. Binary relation learning and classifying for preselection in evolutionary algorithms[J]. IEEE Transactions on Evolutionary Computation, 2020, 24(6): 1125-1139.



### RCPS-MOEA

We extend RCPS to the multi-objective problem~(MOP), accelerating the convergence speed of multi-objective evolutionary algorithms by predicting approximations domination between two solutions through relation models. ([code](), [paper]()) 

> Hao H, Zhou A, Zhang H. An approximated domination relationship based on binary classifiers for evolutionary multiobjective optimization[C]//2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021: 2427-2434.



### REMO

In this work, by designing a relation model to assist the evolutionary algorithm in solving expensive multi-objective optimization problems, the paper proposes an adaptive classification strategy and a voting scoring strategy to enhance the performance of relation models. ([code](), [paper]())

>Hao H, Zhou A, Qian H, et al. Expensive multiobjective optimization by relation learning and prediction[J]. IEEE Transactions on Evolutionary Computation, 2022, 26(5): 1157-1170.


### CREMO

We apply convolutional neural networks~(CNN) to the learning of relation data, effectively overcoming challenges related to data types and dimensions, and demonstrating good performance on continuous, discrete problems as well as mid-to-high dimensional issues. ([code](), [paper]()) 
>Hao H, Zhou A. A Relation Surrogate Model for Expensive Multiobjective Continuous and Combinatorial Optimization[C]//International Conference on Evolutionary Multi-Criterion Optimization. Cham: Springer Nature Switzerland, 2023: 205-217.


### DRSO 

Our observations indicate that updating solely the most favorable solution leads to a reduction in the genetic diversity amongst successive generations, prompting a stagnation of sorts, akin to marching in place. To address this issue, we have devised two relation modeling strategies that entail selecting not only the optimal solution from each generation but also a subset of promising solutions for the update of the population. Leveraging the robust learning and predictive capabilities of the relation model, this approach can significantly enhance the quality of solutions in subsequent generations, thereby elevating the overall efficiency of the algorithm. ([code](), [paper]())


>Hao H, Zhang X Q, Zhou A M. Enhancing SAEAs with unevaluated solutions: a case study of relation model for expensive optimization. Sci China Inf Sci, 2024, 67(2): 120103, doi: 10.1007/s11432-023-3909-x





---

