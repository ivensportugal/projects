import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([[1],[3],[5],[4],[7],[9],[10],[12],[20],[30]], dtype=torch.float)
y = torch.tensor([[1],[9],[25],[16],[49],[81],[100],[144],[400],[900]], dtype=torch.float)

linear = nn.Bilinear(1,1,1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=5e-6)

loss = 1
while(loss > 0.01):
	pred = linear(x,x)
	loss = loss_fn(pred, y)
	print(f'loss: {loss.item():>.3f}')
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()

data = torch.tensor([[15]],dtype=torch.float)
pred = linear(data,data)
print('input: ', data)
print('output: ', pred)