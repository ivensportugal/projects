import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

# Dataset

class CSVDataset(Dataset):

	def __init__(self, filename):
		df = pd.read_csv(filename)
		self.n = df.shape[0]

		Xcol = []
		Ycol = []

		for col in df.columns:
			if 'x' in col: Xcol.append(col)
			else: Ycol.append(col)

		self.X = df[Xcol].to_numpy()
		self.Y = df[Ycol].to_numpy()

	def __len__(self):
		return self.n

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idk = idk.to_list()

		return torch.tensor(self.X[idx,:], dtype=torch.float), torch.tensor(self.Y[idx,:], dtype=torch.float)



training_data = CSVDataset('input11.csv')



# Data loader

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(training_data, batch_size=64)

# for X, y in train_dataloader:
# 	print(X)
# 	print(y)

# Get cpu, gpu or mps device for training.
device = (
	'cuda'
	if torch.cuda.is_available()
	else 'mps'
	if torch.backends.mps.is_available()
	else 'cpu'
)



# Define model
class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_stack = nn.Sequential(
			nn.Linear(1, 1),
			# nn.ReLU()
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_stack(x)
		return logits

model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred = model(X)
		loss = loss_fn(pred, y)
		# print(f'pred: {pred}')
		# print(f'y: {y}')
		# print(f'loss: {loss}')
		# print(f'-------')
		# input()

		# Backpropagation
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if batch % 100 == 0:
			loss, current = loss.item(), (batch+1) * len(X)
			print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}% Avg loss: {test_loss:>8f} \n')



epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done!")

x = torch.tensor([[51]],dtype=torch.float).to(device)
y = model(x)
print(y.item())