# Variance Control for Distributional Reinforcement Learning (ICML 2023)

This folder contains the core code to implement the proposed methods.



## Abstract

Although distributional reinforcement learning (DRL) has been widely examined in the past few years, very few studies investigate the validity of the obtained Q-function estimator in the distributional setting. To fully understand how the estimation errors of the Q-function affect the whole training process, we do some error analysis and theoretically show how to reduce both the bias and the variance of the error terms. With this new understanding, we construct a new estimator Quantiled Expansion Mean (QEM) , and introduce a new DRL algorithm (QEMRL) from the statistical perspective. We extensively evaluate our QEMRL algorithm on a variety of Atari and Mujoco benchmark tasks and demonstrate that QEMRL achieves significant improvement over baseline algorithms in terms of sample efficiency and convergence performance.

## 



## Part 1: Atari

Implementation of Distributional Reinforcement Learning (DRL). The core algorithm of DRL is in `iqn-qrdqn`.

### Package overview

- `iqn-qrdqn/model`: network structures.

- `iqn-qrdqn/agent`:  QR-DQN, IQN,  QEM-DQN, and IQEM-QDN agents.

- `iqn-qrdqn/memory`: basic and prioritized replay buffers.

- `utils`: utility functions.

  

### Examples

You can train QR-DQN agents using:

```
python train_qrdqn.py --cuda 0 --env_id PongNoFrameskip-v4 --seed 0 --config config/qrdqn.yaml
```

You can train QEM-DQN agents using:

```
python train_qem.py --cuda 0 --env_id PongNoFrameskip-v4 --seed 0 --config config/qem.yaml --weight 1.5
```

You can train exploration agents using:

```
python train_qem.py --cuda 0 --env_id PongNoFrameskip-v4 --seed 0 --config config/qem.yaml --weight 1.5 --explo qem(dltv)
```

You can also train IQN agents in the same way.



## Part 2: MuJoCo

Implementation of Distributional Soft Actor Critic (DSAC).  The core algorithm of DSAC is in `rlkit/torch/dsac/`.



### Package overview

- `rlkit/torch`: algos (SAC, DSAC, QEMDSAC)

- `rlkit/torch/networks`:  network structures.

- `rlkit/torch/pytorch_util`: utility functions.

  

### Examples

You can train QEM-DSAC agents using:

```
python qemdsac.py --config configs/dsac-normal-fix-neutral/ant.yaml --gpu 0 --seed 0 --weight 1.25
```

You can train IQEM-DSAC agents using:

```
python qemdsac.py --config configs/dsac-normal-iqn-neutral/ant.yaml --gpu 0 --seed 0 
```

You can also train SAC, DSAC agents in the same way.



## Main packages 

- python 3.6+
- pytorch 1.6+
- gym[all] 0.15+
- mujoco-py 2.0+
- scipy 1.0+
- numpy
- pyyaml



## References

### Thanks to Repos:

- https://github.com/xtma/dsac
- https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch
- https://github.com/rail-berkeley/rlkit
