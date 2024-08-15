import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([[1],[2],[3],[10],[20],[50],[11],[7],[12],[30],[100],[16],[0],[13],[25],[35],[33],[40],[39],[49],
], dtype=torch.float)
y = torch.tensor([[2],[4],[6],[20],[40],[100],[22],[14],[28],[60],[200],[32],[0],[26],[50],[70],[66],[80],[78],[98],
], dtype=torch.float)
# x = torch.tensor([[1],[2]],dtype=torch.float)
# y = torch.tensor([[2],[4]],dtype=torch.float)

linear = nn.Linear(1,1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=1e-3)

loss = 1
while (loss > 0.0001):
	pred = linear(x)
	loss = loss_fn(pred,y)
	# print(linear.weight)
	# print(linear.bias)
	print('loss: ', loss.item())
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()

data = torch.tensor([50],dtype=torch.float)
pred = linear(data)
print('input: ', data.item())
print('output: ', pred.item())
print('loss: ', loss.item())