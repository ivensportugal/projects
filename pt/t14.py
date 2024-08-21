import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([[1],[2],[3], [6], [7], [8], [9]], dtype=torch.float)
y = torch.tensor([[0],[0],[0], [1], [1], [1], [1]], dtype=torch.float)

linear = nn.Sequential(
	nn.Linear(1,1),
	nn.Sigmoid()
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=1e-3)

loss = 1
while(loss > 0.01):
	pred = linear(x)
	loss = loss_fn(pred, y)
	print(f'loss: {loss:>.2f}')
	binary = (pred.detach().numpy() >= 0.5).astype(int)
	accuracy = (binary == y.detach().numpy()).sum()/binary.shape[0] * 100
	print(f'accuracy: {accuracy:>.2f} %')
	loss.backward()
	# for i, layer in enumerate(linear):
	# 	if i == 0:
	# 		print(f'weights: {layer.weight.item():>.2f}')
	# 		print(f'bias: {layer.bias.item():>.2f}')
	# 		print('')
	optimizer.step()
	optimizer.zero_grad()

data = torch.tensor([[2.40]], dtype=torch.float)
pred = linear(data)
print('')
print(f'input: {data.item():>.1f}')
print(f'output: {pred.item():>.1f}')
if pred.item() >= 0.5: print(f'{data.item():>.1f} is above 5.')
else: print(f'{data.item():>.1f} is below 5.')
