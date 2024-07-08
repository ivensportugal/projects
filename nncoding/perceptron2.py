import pandas as pd
import csv

# parameters

filename = 'input.csv'

# hyperparameters
a = 0.01 # learning rate
w = 0 # weight
b = 0 # b

def read_data(filename):

	# read data

	X = []
	Y = []

	with open(filename, mode='r') as file:
		csvFile = csv.reader(file)
		for line in csvFile:
			X.append(int(line[0]))
			Y.append(int(line[1]))

	return X, Y


def train(X, Y):

	w = 0
	b = 0

# actual learning
	while(True):
		gradient_w = 0
		gradient_b = 0
		for i in range(len(X)):

			x = X[i]
			y = Y[i]

			gradient_w = gradient_w + (x*w-y)*x
			gradient_b = gradient_b + (x*w-y)

		w = w - (a*gradient_w)/len(X)
		b = b - (a*gradient_b)/len(X)

		# Loss

		s = 0
		for i in range(len(X)):

			x = X[i]
			y = Y[i]

			s = s + (x*w - y) * (x*w - y)

		L = (s*s)/(2*len(X))

		if (L < 0.01): break

	return [w, b]

def infer(x, model):

	w = model[0]
	b = model[1]

	y = x*w+b
	return y

def print_results(data, prediction, model):

	print(f"The value of w is: {model[0]}")
	print(f"The value of b is: {model[1]}")	
	print(f"For an input {data}, the inference is {prediction}")

def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)
	print_results(data, prediction, model)


if __name__ == "__main__":
	main()