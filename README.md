# DAC
Decentralized adaptive clustering

Here we publish our code for the paper [Decentralized adaptive clustering of deep nets is beneficial for client collaboration](https://arxiv.org/abs/2206.08839) accepted at the [International Workshop on Trustworthy Federated Learning in Conjunction with IJCAI 2022 (FL-IJCAI'22)](https://federated-learning.org/fl-ijcai-2022/).

## Run experiment
To run a label shift experiment on CIFAR-10, use the following line.

```
python main_fed.py --n_rounds 100 --num_clients 100 --local_ep 3 --n_sampled 5 --n_clusters 4 --n_data_train 400 --n_data_val 100 --tau 30 --DAC_var --iid label --dataset cifar10

```