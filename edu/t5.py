import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd

# parameters
filename = 'input3.csv'

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

		self.stack = nn.Sequential(
			nn.Linear(2,1)
		)

	def forward(self, X):
		# self.eval()
		pred = self.stack(X)
		return pred

	def train(self, dataloader):

		super().train()

		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(super().parameters(), lr=1e-3)

		for X, Y in dataloader:
			pred = self(X)
			loss = loss_fn(pred,Y)
			# print(f'train_loss: {loss}')
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

	def test(self, dataloader):

		# self.eval()

		loss_fn = nn.MSELoss()

		loss = 0
		for X, Y in dataloader:
			pred = self(X)
			loss += loss_fn(pred, Y).item()
		loss /= len(dataloader)

		print(f'test_loss: {loss}')

		return loss


def print_results(x, y):
	print(f'For input {x}, the predicted output is {y}.')



def main():

	dataset = CSVDataset(filename)
	dataloader = DataLoader(dataset, batch_size=64)

	model = NeuralNetwork()
	loss = 1
	while(loss > 0.1):
		model.train(dataloader)
		loss = model.test(dataloader)

	data = torch.tensor([[50,50]],dtype=torch.float)
	pred = model(data)

	print_results(data, pred)

if __name__ == '__main__':
	main()