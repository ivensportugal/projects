import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd

# Dataset

class CSVDataset(Dataset):

	def __init__(self, filename):

		self.df = pd.read_csv(filename)
		self.n = self.df.shape[0]
		self.Xcol = []
		self.Ycol = []
		for col in self.df.columns:
			if 'x' in col: self.Xcol.append(col)
			else: self.Ycol.append(col)

	def __len__(self):
		return self.n

	def __getitem__(self, i):
		X = torch.tensor(self.df.loc[i,self.Xcol].to_list(),dtype=torch.float)
		y = torch.tensor(self.df.loc[i,self.Ycol].to_list(),dtype=torch.float)
		return X, y


# Define the model
class NeuralNetwork(nn.Module):

	def __init__(self):
		super().__init__()

		self.loss_fn = nn.MSELoss()
		

		self.stack = nn.Sequential(
			nn.Linear(2,1)
		)

	def train(self, dataloader):

		n = len(dataloader.dataset)
		m = len(dataloader)
		optim = torch.optim.SGD(super().parameters(), lr=1e-4)

		super().train()

		for X, y in dataloader:
			pred = self(X)
			loss = self.loss_fn(pred,y)
			loss.backward()
			optim.step()
			optim.zero_grad()

	def forward(self,X):

		pred = self.stack(X)
		return pred

	def test(self, dataloader):

		n = len(dataloader.dataset)
		m = len(dataloader)

		# super().eval()

		loss = 0
		for X, y in dataloader:

			pred = self(X)
			loss += self.loss_fn(pred, y)

		loss /= m

		print(f'Loss: {loss}')

def print_results(x, y):
	print(f'For input {x}, the predicted output is {y}.')


def main():

	filename = 'input.csv'
	database = CSVDataset(filename)
	dataloader = DataLoader(database, batch_size=64)

	model = NeuralNetwork()
	for i in range(5):
		model.train(dataloader)
	model.test(dataloader)

	data = torch.tensor([[50,7]],dtype = torch.float)
	pred = model(data)
	print_results(data, pred)


if __name__ == '__main__':
	main()