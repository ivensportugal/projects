# Tutorial from
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py

import torch
import torch.nn as nn
import numpy as np

# Create a numpy array.
x = np.array([[1,2],[3,4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()