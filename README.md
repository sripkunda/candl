# candl

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/eb/Candle_flame_by_Shan_Sheehan.jpg" />
</p>

A tiny, pedagogical implementation of a neural network library with a pytorch-like API. The primary use of this library is for education. Use the [actual pytorch](https://github.com/pytorch/pytorch) for more serious deep learning business. 

### Learning

This little project is actually the result of [an article I wrote](https://hackmd.io/@sripkunda/understanding-neural-networks). Using it, you can learn more about how neural networks work and implement everything in candl yourself from scratch.

### Installation 

```shell
pip install candl
```

### Usage

First, import candl.

```python
import candl
```

Candl comes with two modules: `nn` and `Tensor`. The `nn` module contains tools like modules, layers, SGD, MSE, etc. Candl tensors are extensions of numpy ndarrays that can be used to represent data and compute derivatives. 

To train a neural net (let's try to learn XOR), first we can create a model. 

```python
nn = candl.nn

model = nn.Sequential(nn.Linear(2, 10), 
                      nn.ReLU(), 
                      nn.Linear(10, 10), 
                      nn.ReLU(), 
                      nn.Linear(10, 1))
lr = 1e-3

loss_fn = nn.MSE()
optimizer = nn.SGD(model.parameters(), lr)
```

Then, we train: 

```python

data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

for epoch in range(1000):
    for sample in data:
        x = candl.tensor(sample[0])
        y = candl.tensor(sample[1])
        loss = loss_fn(model.forward(x), y)
        loss.backward()
        # The `True` argument automatically zeroes the gradients after a step
        optimizer.step(True) 
```

### Features

- Tensors built upon numpy's ndarrays
- Tensor-valued autograd
- Mean Squared Error Loss Function 
- Stochastic Gradient Descent (SGD) 
- Blocks (Modules) for putting together neural networks 
- Built-in layers: Linear, ReLU, Sigmoid, Tanh
