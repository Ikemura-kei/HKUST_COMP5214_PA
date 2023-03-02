# PA1

## Overview
This PA includes the following models to perform MNIST digit classification.
* KNN: K-Nearest Neighbors
* CNN: LeNet5
* CAN: Context Aggregation Network
* MLP: Multi-Layer Perceptron

## To Run

You can run different experiment in the __run_scripts__ folder, for example:

```
./run_scripts/knn_1.sh
./run_scripts/can_32.sh
```
But first make sure they are executable by:

```
chmod +x ./run_scripts/*
```

Experiment results will be saved to __exp_results/exp_name__ where __exp_name__ is specified in the bash scripts for each experiment.

## Note

No distributed training is implemented, but will automatically use GPU0, you can modify this behavior by changing the code in __main.py__.