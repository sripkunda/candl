import numpy as np
from candl.tensor import Tensor

class Parameter: 
  def __init__(self, data, requires_grad=True):
    self.data = Tensor(data, requires_grad, float) # Set data of parameter

class Module: 
  def __init__(self, *args): 
    self.__children = args or [] # Initialize module with given arguments as children
        
  def forward(self, x):
    # Call .forward() on all of the children
    for child in self.__children:
      x = child.forward(x)
    return x

  def parameters(self):
    # Fetch all of the parameters by going through each child and getting all of the Parameter instances
    parameters = []
    for child in self.__children:
      for (key, value) in child.__dict__.items():
        if isinstance(value, Parameter):
          parameters.append(value)
    return parameters
      
# Just to make it more pytorchy, allows you to chain together modules to make a "model" (or just an arbitrary chain of modules)
def Sequential(*args): 
  return Module(*args)

# Just a basic linear layer with an optional bias node
class Linear(Module):
  def __init__(self, input_neurons, output_neurons, bias=True):
    self.weights = Parameter(np.random.randn(input_neurons, output_neurons))
    self.bias = bias
    if bias:
      self.biases = Parameter(np.random.randn(output_neurons))

  def forward(self, x):
    out = x.matmul(self.weights.data)
    if self.bias: 
      return out + self.biases.data
    return out
  
class ReLU(Module):
  def forward(self, x):
    return x.relu()

class Sigmoid(Module):
  def forward(self, x):
    return x.sigmoid()

class Tanh(Module):
  def forward(self, x):
    return x.tanh()

class SGD: 
  def __init__(self, parameters, learning_rate):
    self.parameters = parameters
    self.learning_rate = learning_rate

  def step(self, zero_grad=False):
    for param in self.parameters: 
      param = param.data # Get the data inside the parameter
      # w_i+1 <- w_i - lr * (dL/dw_i)
      param.els = param.els - self.learning_rate * param.grad
      # Reset the gradient if specified
      if zero_grad:
        param.zero_grad()
    
  def zero_grad(self): 
    for param in self.parameters: 
      param.zero_grad()

def MSE(): 
  # Compute the mean squared error loss given a prediction (yhat) and a target (y)
  def compute_loss(yhat, y):    
    N = np.prod(y.els.shape, dtype=float)
    return ((y - yhat) ** 2).sum() / N
  return compute_loss