import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
# import numpy as np

# parameters
filename = 'input.csv'

# Dataset

class CSVDataset(Dataset):

	def __init__(self, filename):

		# super.__init__()
		self.data = pd.read_csv(filename)
		self.n = self.data.shape[0]
		self.Xcol = []
		self.Ycol = []
		for col in self.data.columns:
			if 'x' in col: self.Xcol.append(col)
			else: self.Ycol.append(col)

	def __len__(self):
		return self.n

	def __getitem__(self, index):
		X = torch.tensor(self.data.loc[index,self.Xcol].to_list(),dtype=torch.float)
		y = torch.tensor(self.data.loc[index,self.Ycol].to_list(),dtype=torch.float)
		return X, y


# Define model
class NeuralNetwork(nn.Module):

	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(2,1)
			)

	def forward(self, X):
		return self.network(X)

	def train(self, dataloader):

		n = len(dataloader.dataset)
		super().train()

		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(super().parameters(), lr=1e-4)

		for batch, (X,y) in enumerate(dataloader):

			pred = self(X)
			loss = loss_fn(pred,y)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

	def test(self, dataloader):

		# super().eval()

		loss_fn = nn.MSELoss()

		loss = 0
		for X, y in dataloader:
			pred = self(X)
			loss += loss_fn(pred,y).item()
		loss /= dataloader.batch_size
		print(f'loss: {loss}')



def print_results(x, y):
	print(f'For an input {x}, the output is {y}')





def main():

	dataset = CSVDataset(filename)
	dataloader = DataLoader(dataset, batch_size=64)
	model = NeuralNetwork()
	model.train(dataloader)
	model.test(dataloader)

	data = torch.tensor([[50,51]],dtype=torch.float)
	pred = model(data)

	print_results(data, pred)




if __name__ == '__main__':
	main()