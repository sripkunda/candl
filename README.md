# candle

A tiny, pedagogical implementation of a neural network library with a pytorch-like API. The primary use of this library is for education. Use the [actual pytorch](https://github.com/pytorch/pytorch) for more serious deep learning business. 

## learning

This project is actually a result of [an article I wrote](https://hackmd.io/@sripkunda/understanding-neural-networks). Using it, you can learn more about how neural networks work and also implement everything in candle yourself from scratch.

## features

- Tensors built upon numpy's ndarrays
- Tensor-valued autograd
- Mean Squared Error Loss Function 
- Stochastic Gradient Descent (SGD) 
- Blocks (Modules) for putting together neural networks 
- Built-in layers: 
    - Linear
    - ReLU Activation Layer
    - Sigmoid Activation Layer
    - Tanh Activation Layer