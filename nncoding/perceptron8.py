# architecture n - 1
# feature: vectorization

import pandas as pd

# parameters
filename = 'input.csv'

W = []
B = []

# hyperparameter
l = 0.01 # learning rate

def read_data(filename):

	df = pd.read_csv(filename)

	X = df.x
	Y = df.y

	return X, Y

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(X)


	print(f'X: {X}')
	print(f'W.values: {W.values}')
	# print(f'B: {B}')
	# print(f'Y: {Y}')
	print(f'X*W: {X.values*W.values}')

	S = ((X.values*W.values+B.values)-Y.values)
	
	loss = S.transpose().dot(S) / (2*n)

	return loss.squeeze()

def train(X, Y):

	n = len(X)

	global W
	global B

	W = pd.DataFrame(0.1,range(n),['w'])
	B = pd.DataFrame(0.1,range(n),['b'])

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.1: break

		grad_W = (infer(X) - Y.values).transpose.dot(X)
		grad_B = (infer(X) - Y.values)

		W = W.values - l*grad_W
		B = B.values - l*grad_B

	return [W,B,loss]


def infer(X):

	Y = X.transpose().dot(W) + B.sum()
	return Y

def print_results(x, y, model):

	print(f'Weights: {model[0]}')
	print(f'Biases: {model[1]}')
	print(f'Loss: {model[2]}')

	print(f'For an input {x}, the predicted output is {y}.')

def main():

	X, Y  = read_data(filename)
	model = train(X, Y)

	data = [50, 51]
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()