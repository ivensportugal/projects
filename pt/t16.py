import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# read datasets

result = pd.read_csv('f1/results_history.csv')
pitstop = pd.read_csv('f1/pitstops.csv')

df = pd.merge(result, pitstop, how='inner', left_on=['Season','CircuitID','DriverID'], right_on=['season','circuitId','driverId'])

# df = df[['season','circuitId','driverId','Grid','stop','lap','duration','Position']]
df = df[['season','circuitId','driverId','Grid','stop','Position']]
df = df.groupby(by=['season','circuitId','driverId','Grid','Position']).max().reset_index()

# some filters before training
df = df[['circuitId','Grid','stop','Position']]
df = df[df.Position == 1]

# handling of categorical data (encoding)
categories = pd.DataFrame([[j,i] for i, j in enumerate(pd.unique(df['circuitId']))], columns=['circuitId','circuitId_'])
df = pd.merge(df, categories, how='left', on='circuitId')

x = torch.tensor(df[['circuitId_', 'Grid', 'stop']].to_numpy(), dtype=torch.float)
y = torch.tensor(df[['Position']].to_numpy(), dtype=torch.float)

r  = torch.arange(len(x)).reshape(-1,1)
y_ = torch.zeros(len(x), 24, dtype=torch.float)

y_[r,y.type(torch.int)-1] = 1

linear = nn.Sequential(
	nn.Linear(3,24),
	nn.Linear(24,24),
	nn.Linear(24,24),
	nn.Linear(24,24),
	nn.ReLU()
)
linear2 = nn.Softmax(dim=1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=1e-4)

accuracy = 0
i = 1
while accuracy < 0.99:

	logits = linear(x)
	pred = linear2(logits)
	loss = loss_fn(logits, y_)
	print(f'loss: {loss:>.2f}')

	accuracy = (pred.argmax(dim=1, keepdim=True)+1 == y).type(torch.float).sum().item() / len(y)
	print(f'accuracy: {accuracy*100:>.2f}%')

	loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	# if i%100 == 0:
	# 	print('logts')
	# 	print(logits[0])
	# 	print('y_')
	# 	print(y_[0])
	# 	input()

	i = i+1

