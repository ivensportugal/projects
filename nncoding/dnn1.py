# architecture: 2 - 2 - 1
# feature: vectorization

import pandas as pd
import numpy as np

# parameters
filename = 'input2.csv'

# hyperparameters

l = 0.001 # learning rate

Wl1 = []
Bl1 = []

Wl2 = []
Bl1 = []

WL = []
bL = []


def read_data(filename):

	df = pd.read_csv(filename)

	Xcols = []
	Ycols = []
	for col in df.columns:
		if 'x' in col: Xcols.append(col)
		else: Ycols.append(col)

	X = df[Xcols].to_numpy()
	Y = df[Ycols].to_numpy()

	return X, Y

def activate(Z):
	return Z

def calculate_loss(X, Y):

	n = len(Y)

	Xt = np.linalg.matrix_transpose(X)

	Zl1 = Wl1 @ Xt + Bl1
	Al1 = activate(Zl1)

	Zl2 = Wl2 @ Al1 + Bl2
	Al2 = activate(Zl2)

	ZL = WL @ Al2 + bL
	AL = activate(ZL)

	Yi = np.linalg.matrix_transpose(AL)

	T = Yi - Y
	loss = (np.linalg.matrix_transpose(T)@T)[0][0] / (2*n)

	return loss


def train(X, Y):

	n = X.shape[0]
	k = X.shape[1]

	global Wl1, Wl2, WL
	global Bl1, Bl2, bL

	Wl1 = np.full((k,k), 0.1)
	Bl1 = np.full((k,1), 0.1)

	Wl2 = np.full((k,k), 0.1)
	Bl2 = np.full((k,1), 0.1)

	WL = np.full((1,k), 0.1)
	bL = np.full((1,1), 0.1)


	Xt = np.linalg.matrix_transpose(X)

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.001: break

		Zl1 = Wl1 @ Xt + Bl1
		Al1 = activate(Zl1)

		Zl2 = Wl2 @ Al1 + Bl2
		Al2 = activate(Zl2)

		ZL = WL @ Al2 + bL
		AL = activate(ZL)

		Yi = np.linalg.matrix_transpose(AL)

		Yd = Yi - Y
		One = np.ones((1,n))

		grad_WL = np.linalg.matrix_transpose(Al2 @ Yd) / n
		grad_bL = np.linalg.matrix_transpose(One @ Yd) / n

		grad_Wl2 = np.linalg.matrix_transpose(Al1 @ Yd @ WL) / n
		grad_Bl2 = np.linalg.matrix_transpose(One @ Yd @ WL) / n

		grad_Wl1 = np.linalg.matrix_transpose(Xt @ Yd @ WL @ Wl2) / n
		grad_Bl1 = np.linalg.matrix_transpose(One @ Yd @ WL @ Wl2) / n

		WL = WL - l*grad_WL
		bL = bL - l*bL

		Wl2 = Wl2 - l*grad_Wl2
		Bl2 = Bl2 - l*grad_Bl2

		Wl1 = Wl1 - l*grad_Wl1
		Bl1 = Bl1 - l*grad_Bl1

	return [[Wl1,Bl1], [Wl2,Bl2], [WL, bL], loss]

def infer(X, model):

	Xt = np.linalg.matrix_transpose(X)

	Zl1 = model[0][0] @ Xt + model[0][1]
	Al1 = activate(Zl1)

	Zl2 = model[1][0] @ Al1 + model[1][1]
	Al2 = activate(Zl2)

	ZL = model[2][0] @ Al2 + model[2][1]
	AL = activate(ZL)

	Y = AL

	return Y

def print_results(X, Y, model):

	print(f'Hidden Layer 1: {model[0]}')
	print(f'Hidden Layer 2: {model[1]}')
	print(f'Output Layer: {model[2]}')

	print(f'Loss: {model[3]}')

	print(f'For an input {X}, the predicted output is {Y}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = [[50, 51]]
	prediction = infer(data, model)

	print_results(data, prediction, model)


if __name__ == '__main__':
	main()