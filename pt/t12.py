import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([[1],[3],[5],[4],[7],[9],[10],[12],[20],[30]], dtype=torch.float)
y = torch.tensor([[1],[9],[25],[16],[49],[81],[100],[144],[400],[900]], dtype=torch.float)

linear1 = nn.Linear(1,1)
linear2 = nn.Linear(1,1)

loss_fn = nn.MSELoss()
optimizer1 = torch.optim.SGD(linear1.parameters(), lr=1e-9)
optimizer2 = torch.optim.SGD(linear2.parameters(), lr=1e-)

loss = 1
while(loss > 0.1):
	pred1 = linear1(x)
	pred2 = linear2(pred1)
	loss = loss_fn(pred2, y)
	# print('linear1.weight: ',linear1.weight)
	# print('linear1.bias: ',linear1.bias)
	# print('linear2.weight: ',linear2.weight)
	# print('linear2.bias: ',linear2.bias)
	print('loss: ', loss.item())
	loss.backward()
	optimizer1.step()
	optimizer2.step()
	optimizer1.zero_grad()
	optimizer2.zero_grad()

data = torch.tensor([[2]],dtype=torch.float)
pred = linear(data)
print('input: ', data)
print('output: ', pred)