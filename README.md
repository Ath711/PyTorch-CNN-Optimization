# PyTorch CNN Optimization

This repository contains a simple Convolutional Neural Network (CNN) implemented in PyTorch for the MNIST dataset. The code provides a baseline implementation and demonstrates several optimization strategies to improve training efficiency.

## Table of Contents

1. [Introduction](##introduction)
2. [Dependencies](##dependencies)
3. [Model Architecture](##model-architecture)
4. [Dataset](##dataset)
5. [Training](##training)
    1. [Optimization Strategies](###optimization-strategies)
6. [Results](##results)
    1. [Original Implementation](###original-implementation)
    2. [Optimized Implementation](###optimized-implementation)
7. [How to Use](##how-to-use)

## Introduction

The primary goal of this project is to showcase a basic CNN implementation for the MNIST dataset and explore various optimization strategies to enhance training speed and efficiency.

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- tensorboard
- torch.profiler

You can install the required packages using the following:

```bash
pip install torch torchvision tensorboard torchprofiler
```

## Model Architecture

The model used is a simple CNN with one convolutional layer, a ReLU activation function, and a fully connected layer. The architecture is defined in `SimpleCNN` within the provided script.

## Dataset

The MNIST dataset is utilized for training and validation. It is loaded using torchvision's `datasets.MNIST` and split into training and validation sets.

## Training

### Optimization Strategies

The code includes various optimization strategies:

- Multi-process Data Loading
- Memory Pinning
- Increase Batch Size
- Reduce Host to Device Copy
- Set Gradients to None
- Automatic Mixed Precision (AMP)
- Train in Graph Mode

These strategies aim to improve data loading efficiency, reduce memory overhead, and accelerate training.

## Results

### Original Implementation

The original implementation logs training accuracy using TensorBoard in the 'logs/original' directory.

### Optimized Implementation

The optimized implementation incorporates the mentioned strategies and logs training loss and accuracy in the 'logs/optimized' directory.

## How to Use

1. **Clone this repository:**

    ```bash
    git clone https://github.com/yourusername/pytorch-cnn-optimization.git
    cd pytorch-cnn-optimization
    ```

2. **Install dependencies:**

    ```bash
    pip install torch torchvision tensorboard torchprofiler
    ```

3. **Experiment with optimization strategies:**

    ```bash
    python optimized_training.py
    ```

4. **View TensorBoard logs:**

    ```bash
    tensorboard --logdir=logs
    ```

    Visit [http://localhost:6006](http://localhost:6006) in your browser to visualize training metrics.


