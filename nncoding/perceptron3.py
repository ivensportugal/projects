import pandas as pd
import csv

# parameters
filename = 'input.csv'

# hyperparameters
a = 0.1 #learning rate
W = [0,0] # initial weights
B = [0,0] # initial biases

def read_data(filename):

	df = pd.read_csv(filename)
	X = df['n']
	Y = df['n1']

	return X, Y

def calculate_loss(X, Y, model):

	n = len(X)

	s = 0
	for i in range(n):
		t = X[i]*model[0][0] + model[1][0] + X[i]*model[0][1] + model[1][1] - Y[i]
		s = s + t*t

	loss = s/(2*n)

	return loss

def train(X, Y):

	n = len(X)

	print(f'calculated loss: {calculate_loss(X, Y, [W, B])}')

	while(calculate_loss(X, Y, [W, B]) > 0.01):

		s = 0
		for i in range(n):
			s = s + (X[i]*W[0] + B[0] + X[i]*W[1] + B[1] - Y[i])*X[1]
		grad_w1 = s/(2*n)
		grad_w2 = s/(2*n)

		W[0] = W[0] - a * grad_w1
		W[1] = W[1] - a * grad_w2

		s = 0
		for i in range(n):
			s = s + (X[i]*W[0] + B[0] + X[i]*W[1] + B[1] - Y[i])
		grad_b1 = s/(2*n)
		grad_b2 = s/(2*n)

		B[0] = B[0] - a * grad_b1
		B[1] = B[1] - a * grad_b2

		print(f'calculated loss2: {calculate_loss(X, Y, [W, B])}')
		input()

	return [W, B]

def infer(x, model):
	return x * model[0][0] + model[1][0] + x * model[0][1] + model[1][1]

def print_results(x, y, model, X, Y):

	print(f"Weights: {W[0]}, {W[1]}")
	print(f"Biases:  {B[0]}, {B[1]}")

	print(f"Loss: {calculate_loss(X, Y, model)}")

	print(f"For an input {x}, the prediction is {y}.")


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)

	print_results(data, prediction, model, X, Y)
	

if __name__ == '__main__':
	main()