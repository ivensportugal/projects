import torch
import torch.nn as nn
import numpy as np

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b

print(type(y))
print(y)

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)