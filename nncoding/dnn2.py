# architecture: anything
# feature: modularization

import pandas as pd
import numpy as np

# parameters
filename = 'input2.csv'

class Layer:

	m = 0
	l = 0 # learning_rate

	w = 0 # initial weight values

	Wl = []
	Bl = []

	# most recent execution
	X = []
	Y = []


	def __init__(self, m, l=0.0001, w=0.1, b=0.1):

		self.m = m
		self.l = l
		self.w = w

		self.Bl = np.full((m,1),b)

	def setWeights(self, k):
		self.Wl = np.full((self.m,k),self.w)

	def activate(self, Z):
		return Z

	def grad_activate(self):
		return 1

	def output(self, X):

		self.X = X

		Zl = self.Wl @ X + self.Bl
		Al = self.activate(Zl)
		Y = Al

		self.Y = Y

		return Y

# class InputLayer(Layer):

# 	def __init__(self, m):
# 		super().__init__(m, 0)

# 	def output(self, X):
# 		Xt = np.linalg.matrix_transpose(X)
# 		return Xt

class DenseLayer(Layer):

	def __init__(self, m, l=None, w=None, b=None):

		if l == None and w == None and b == None: super().__init__(m)
		if l == None and w == None and b != None: super().__init__(m, b=b)
		if l == None and w != None and b == None: super().__init__(m, w=w)
		if l == None and w != None and b != None: super().__init__(m, w=w, b=b)
		if l != None and w == None and b == None: super().__init__(m, l=l)
		if l != None and w == None and b != None: super().__init__(m, l=l, b=b)
		if l != None and w != None and b == None: super().__init__(m, l=l, w=w)
		if l != None and w != None and b != None: super().__init__(m, l=l, w=w, b=b)

class OutputLayer(Layer):

	def __init__(self, m, l=None, w=None, b=None):

		if l == None and w == None and b == None: super().__init__(m)
		if l == None and w == None and b != None: super().__init__(m, b=b)
		if l == None and w != None and b == None: super().__init__(m, w=w)
		if l == None and w != None and b != None: super().__init__(m, w=w, b=b)
		if l != None and w == None and b == None: super().__init__(m, l=l)
		if l != None and w == None and b != None: super().__init__(m, l=l, b=b)
		if l != None and w != None and b == None: super().__init__(m, l=l, w=w)
		if l != None and w != None and b != None: super().__init__(m, l=l, w=w, b=b)

	def output(self, A):

		Yt = super().output(A)
		Y = np.linalg.matrix_transpose(Yt)

		return Y

class NeuralNetwork:

	layers = []
	loss = 0

	def __init__(self,layers):

		k = 0
		for l in layers:
			l.setWeights(k)
			k = l.m

		self.layers = layers


	def calculate_loss(self, X, Y):

		n = X.shape[0]

		Yi = self.infer(X)
		T = Yi - Y
		loss = np.linalg.matrix_transpose(T) @ T / (2*n)

		return loss[0][0]

	def train(self, X, Y):

		n = X.shape[0]
		k = X.shape[1]

		self.layers[0].setWeights(k)

		while(True):

			loss = self.calculate_loss(X, Y)
			print(f'loss: {loss}')
			self.loss = loss
			if loss < 0.01: break

			Yi = self.infer(X)
			Yd = Yi - Y

			One = np.ones((1,n))

			W = np.identity(1)
			for l in reversed(self.layers):
				grad_W = np.linalg.matrix_transpose(l.X @ Yd @ W)
				grad_B = np.linalg.matrix_transpose(One @ Yd @ W)

				W = W @ l.Wl

				l.Wl = l.Wl - l.l*grad_W
				l.Bl = l.Bl - l.l*grad_B

	def infer(self, X):

		X = np.linalg.matrix_transpose(X)

		i = 0
		for l in self.layers:
			X = l.output(X)
		return X

def read_data(filename):

	df = pd.read_csv(filename)

	Xcol = []
	Ycol = []
	for col in df.columns:
		if 'x' in col: Xcol.append(col)
		else: Ycol.append(col)

	X = df[Xcol].to_numpy()
	Y = df[Ycol].to_numpy()

	return X, Y

def print_results(x, y, model):
	
	for l in model.layers:
		print(f'Layer - W: {l.Wl}')
		print(f'Layer - B: {l.Bl}')
		print('')

	print(f'Loss: {model.loss}\n')

	print(f'For input {x}, the predicted output is {y}.')

def main():

	X, Y = read_data(filename)

	hl1 = DenseLayer(2,0.00001)
	hl2 = DenseLayer(2,0.00001)
	hl3 = DenseLayer(2,0.00001)
	hl4 = DenseLayer(2,0.00001)
	ol = OutputLayer(1)

	model = NeuralNetwork([hl1,hl2,hl3,hl4,ol])
	model.train(X, Y)

	data = [[50,51]]
	prediction = model.infer(data)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()