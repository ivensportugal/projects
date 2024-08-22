import torch
import torch.nn as nn
import numpy as np

# x = torch.tensor([[0],[4],[7],[2],[8],[5],[2.5],[3.5],[4.5],[5.5]], dtype=torch.float)
# y = torch.tensor([[0],[1],[2],[1],[2],[1],[0],[1],[1],[1]], dtype=torch.float)

x = torch.tensor([[2.8],[2.9],[9],[1],[10],[6],[4],[7],[1],[2],[2],[10],[5],[7],[4],[8],[0],[0],[8],[3],[10],[7],[6],[0],[4],[1],[9],[1],[6],[10],[6],[5],[6],[9],[1],[3],[0],[7],[2],[8],[8],[6],[4],[7],[2],[3],[1],[6],[8],[7],[4],[2],[3],[2],[5],[6],[3],[7],[1],[8],[2],[10],[8],[5],[2],[3],[10],[9],[7],[7],[10],[2],[4],[8],[9],[0],[7],[8],[2],[4],[10],[9],[10],[9],[1],[7],[6],[6],[7],[10],[0],[9],[2],[3],[10],[2],[3],[5],[10],[7],[9],[4]], dtype=torch.float)
y = torch.tensor([[0],[0],[2],[0],[2],[2],[1],[2],[0],[0],[0],[2],[1],[2],[1],[2],[0],[0],[2],[1],[2],[2],[2],[0],[1],[0],[2],[0],[2],[2],[2],[1],[2],[2],[0],[1],[0],[2],[0],[2],[2],[2],[1],[2],[0],[1],[0],[2],[2],[2],[1],[0],[1],[0],[1],[2],[1],[2],[0],[2],[0],[2],[2],[1],[0],[1],[2],[2],[2],[2],[2],[0],[1],[2],[2],[0],[2],[2],[0],[1],[2],[2],[2],[2],[0],[2],[2],[2],[2],[2],[0],[2],[0],[1],[2],[0],[1],[1],[2],[2],[2],[1]], dtype=torch.float)

r = torch.arange(len(x)).reshape(-1,1)
y_ = torch.zeros(len(x), 3, dtype=torch.float)
y_[r,y.type(torch.int)] = 1


linear = nn.Bilinear(1,1,3)
linear2 = nn.Softmax(dim=1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=1e-4)

accuracy = 0
while accuracy < 0.9:
	
	logits = linear(x,x)
	pred = linear2(logits)
	# print('logits')
	# print(logits)
	# print('pred')
	# print(pred)
	# print('y_')
	# print(y_)
	loss = loss_fn(logits, y_)
	print(f'loss: {loss:>.2f}')

	accuracy = (pred.argmax(dim=1, keepdim=True) == y).type(torch.float).sum().item() / len(y)
	# for i in zip(pred.argmax(dim=1, keepdim=True), y):
	# 	print(f'{i[0].item()} = {i[1].item()}')
	print(f'accuracy: {accuracy*100} %')
	# input()

	# for i, l in enumerate(linear):
	# 	if i==0:
	# 		print(f'loss:{loss:>.1f} acc:{accuracy*100:>.1f}', end=' ')
	# 		for j, k in enumerate(zip(l.weight,l.bias)):
	# 			print(f'w{j}:{k[0].item():>.1f} b{j}: {k[1].item():>.1f}', end=' ')
	# 		# print('')

	loss.backward()

	# for i, l in enumerate(linear):
	# 	if i==0:
	# 		print(f'{l.weight.grad}', end=' ')

	# for i in range(len(y)):
	# 	print(f'{(pred.argmax(dim=1, keepdim=True))[i].item()} - {y[i].item()}')

	optimizer.step()
	optimizer.zero_grad()

data = torch.tensor([[1],[0],[2.3],[4.3],[6.3],[9.3]], dtype=torch.float)
pred = linear(data, data)
pred = linear2(pred)
print(pred)
classes = pred.argmax(dim=1, keepdim=True).type(torch.int)
print('')
print(f'input: {data}')
print(f'output: {classes}')