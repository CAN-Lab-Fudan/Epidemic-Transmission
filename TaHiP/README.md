# Predicting Higher-Order Dynamics With Unknown Hypergraph Topology

This repo is the official Pytorch implementation of ["Predicting Higher-Order Dynamics With Unknown Hypergraph Topology"](https://ieeexplore.ieee.org/document/10794522). In this work, we proposed a two-step method called the topology-agnostic higher-order dynamics prediction (TaHiP) algorithm. The observations of nodal states of the target hypergraph are used to train a surrogate matrix, which is then employed in the dynamical equation to predict future nodal states in the same hypergraph, given the initial nodal states. Furthermore, experiments in synthetic and real-world hypergraphs show that the prediction error of the TaHiP algorithm increases with mean hyperedge size of the hypergraph, and could be reduced if the hyperedge size distribution of the hypergraph is known.

## Supported Versions

``3.8 <= Python <= 3.13`` is required.

### Prerequisites

numpy; matplotlib; scikit-learn; torch==1.13.0



## Running the code
- hypergraph_formation from exsisting datasets 

`python run hypergraph_formation.py`

- training and forecasting

`python run_product_model_copredict.py`




## Citation

If you find this repo useful, please cite our paper. 

```
@ARTICLE{10794522,
  author={Zhou, Zili and Li, Cong and Mieghem, Piet Van and Li, Xiang},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers}, 
  title={Predicting Higher-Order Dynamics With Unknown Hypergraph Topology}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Power system dynamics;Topology;Heuristic algorithms;Mathematical models;Predictive models;Accuracy;Prediction algorithms;Vectors;Integrated circuit modeling;Training;Nonlinear system;dynamics on networks;predicting higher-order dynamics;contagion;hypergraph},
  doi={10.1109/TCSI.2024.3513406}}
```

## Acknowledgement

We appreciate Professor Austin R. Benson and his team a lot for their valuable hypergraph datasets provided in https://www.cs.cornell.edu/~arb/data/
