import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
# import numpy as np

# parameters
filename = 'input9.csv'

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
			nn.Linear(1,1),
			# nn.Softmax(dim=0)
			nn.Sigmoid()
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
			print(f'loss: {loss}')
			print('training')
			for i, p in enumerate(X):
				# print(f'X: {X[i]}')
				print('X[i]: ',end='')
				for a in X[i]:
					print(f'{a.item():>.3f}', end=' ')
				print('')
				# print(f'pred: {pred[i]}')
				print('p[i]: ',end='')
				for a in pred[i]:
					print(f'{a.item():>.3f}', end=' ')
				print('')
				# print(f'Y: {Y[i]}')
				print('Y[i]: ',end='')
				for a in Y[i]:
					print(f'{a.item():>.3f}', end=' ')
				print('')
				print(f'loss: {loss}')
				input()
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
	print(f'input: {x.detach().numpy()}')
	print(f'output: {y.detach().numpy()}')
	i = y.argmax(1).item()
	print(f'i:{i}')
	# print(f'result lies between {i*10} and {i*10+10}.')



def main():

	dataset = CSVDataset(filename)
	dataloader = DataLoader(dataset, batch_size=64)

	model = NeuralNetwork()
	epoch=300
	for i in range(epoch):
		print(f'epoch: {i}',end=' ')
		model.train(dataloader)
		model.test(dataloader)

	# for x, y in dataloader:
	# 	for i in range(len(x)):
	# 		pred = model(x[i])
	# 		# print_results(x,pred)
	# 		print(f'expected: {y[i].argmax()}')
	# 		print(f'obtained: {pred.argmax()}')
	# 		input()
	# print('--')
	data = torch.tensor([[0]],dtype=torch.float)
	pred = model(data)
	print_results(data, pred)

if __name__ == '__main__':
	main()