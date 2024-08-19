import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd

# parameters
filename = 'input10.csv'

class CSVDataset(Dataset):

	def __init__(self, filename):

		self.df = pd.read_csv(filename)

		self.Xcol = []
		self.Ycol = []

		for col in self.df.columns:
			if 'x' in col: self.Xcol.append(col)
			else: self.Ycol.append(col)

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, i):

		X = torch.tensor(self.df.loc[i,self.Xcol].to_list(),dtype=torch.float)
		Y = torch.tensor(self.df.loc[i,self.Ycol].to_list(),dtype=torch.float)

		return X, Y


class NeuralNetwork(nn.Module):

	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.stack = nn.Bilinear(1, 1, 1)

	def forward(self, X):
		X = self.flatten(X)
		pred = self.stack(X, X)
		return pred

	def train(self, dataloader):

		super().train()

		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(super().parameters(), lr=1e-9)

		for X, Y in dataloader:

			pred = self(X)
			loss = loss_fn(pred,Y)
			print(f'loss: {loss.item():>.2f}')

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

def print_results(x, y):
	print(f'input: {x}')
	print(f'output: {y}')

def main():
	dataset = CSVDataset(filename)
	dataloader = DataLoader(dataset, batch_size=64)

	model = NeuralNetwork()
	epoch = 20
	for i in range(epoch):
		print(f'epoch: {i}')
		model.train(dataloader)

	data = torch.tensor([[10]],dtype=torch.float)
	pred = model.forward(data)
	print_results(data, pred)

if __name__ == '__main__':
	main()