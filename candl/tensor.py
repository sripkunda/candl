import numpy as np

class Tensor:
  def __init__(self, els, requires_grad=False, dtype=None, tape=None):
    # Store the elements of our tensor in a numpy ndarray
    self.els = np.array(els, dtype=dtype)
    # Decide whether or not we are computing gradients
    self.requires_grad = requires_grad
    # Initialize an empty gradient tape for autodiff
    self.tape = tape or []
    # If the dtype is not "float", raise an error
    if not self.els.dtype == float and requires_grad:
        raise TypeError("Autodiff is only allowed for tensors of type \"float\"!")
    # Otherwise, initialize our gradient
    if requires_grad:
        self.zero_grad()
    else: 
      self.grad = None

  def __getitem__(self, n):
    return self.els[n]

  def __repr__(self):
    return self.els.__str__()

  def __op(self, other, operation, derivative_a, derivative_b=None):
    # Turn the other argument into a tensor if it isn't already one
    if not isinstance(other, Tensor):
      other = Tensor([other], dtype=self.els.dtype, requires_grad=False)
    # If either one of our tensors require grad, then the output tensor also requires grad
    requires_grad = self.requires_grad or other.requires_grad
    # Find the output tensor by executing the given operation
    out = Tensor(operation(self.els, other.els), requires_grad=requires_grad, tape=(other.tape + self.tape))
    # Push a function into the tape that we can call during reverse mode
    def tape_action(): 
      # Execute the derivative functions for the operation if the gradients exist
      if self.grad is not None: 
        self.grad = self.grad + derivative_a(self.els, other.els, out.grad)
        if self.grad.shape != self.els.shape:
          self.grad = np.add.reduce(self.grad)
      if other.grad is not None and derivative_b is not None:
        other.grad = other.grad + derivative_b(self.els, other.els, out.grad)
        if other.grad.shape != other.els.shape:
          other.grad = np.add.reduce(other.grad)
    out.tape.append(tape_action)
    # Return the outputted value
    return out

  def __add__(self, other):
    return self.__op(other, lambda a, b: a + b, lambda a, b, dc: dc, lambda a, b, dc: dc)

  def __sub__(self, other):
    return self.__op(other, lambda a, b: a - b, lambda a, b, dc: dc, lambda a, b, dc: -dc)

  def __rsub__(self, other):
    return -self + other

  def __mul__(self, other):
    return self.__op(other, lambda a, b: a * b, lambda a, b, dc: b * dc, lambda a, b, dc: a * dc)

  def __pow__(self, other): 
    return self.__op(other, lambda a, b: a ** b, lambda a, b, dc: b * a ** (b - 1) * dc, lambda a, b, dc: np.log(a) * a ** b * dc)

  def __neg__(self): 
    return self * -1

  def __truediv__(self, other):
    return self * other ** -1

  def __rtruediv__(self, other):
    return self ** -1 * other

  def sum(self):
    return self.__op(1, lambda a, b: np.sum(a), lambda a, b, dc: dc)

  def relu(self): 
    return self.__op(1, lambda a, b: a * (a > 0), lambda a, b, dc: (a > 0) * dc)

  def sigmoid(self): 
    sigmoid = lambda a, b: 1 / (1 + np.exp(-a))
    return self.__op(1, sigmoid, lambda a, b, dc: sigmoid(a, b) * (1 - sigmoid(a, b)) * dc)

  def tanh(self): 
    tanh = lambda a, b: np.tanh(a)
    return self.__op(1, tanh, lambda a, b, dc: 1 - tanh(a, b) ** 2 * dc)

  def matmul(self, other):
    return self.__op(other, lambda a, b: np.dot(a, b), lambda a, b, dc: np.dot(dc, b.T), lambda a, b, dc: np.dot(a.T, dc)) 

  def backward(self): 
    self.grad = np.ones(self.els.shape) # Set the seed
    while len(self.tape) > 0: 
      self.tape.pop()() # Remove and execute the last element of the tape
    return self

  def zero_grad(self): 
    self.grad = np.zeros_like(self.els)

  __radd__ = __add__
  __rmul__ = __mul__