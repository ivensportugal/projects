import pandas as pd

# parameters
filename = 'input2.csv'

# hyperparameters
a = 0.1 # learning rate
W = [0,0] # weights
B = [0,0] # biases

def read_data(filename):

	df = pd.read_csv(filename)
	X1 = df['x1'].to_list()
	X2 = df['x2'].to_list()
	X = [[X1[i],X2[i]] for i in range(len(X1))]
	Y = df['y'].to_list()

	return X, Y

def calculate_loss(X, Y, model):

	n = len(X)

	s = 0
	for i in range(len(X)):
		t = X[i][0] * W[0] + B[0] + X[i][1] * W[1] + B[1] - Y[i]
		s = s + t*t

	loss = s/(2*n)

	return loss


def train(X, Y):

	n = len(X)

	while(calculate_loss(X, Y, [W, B]) > 0.00001):

		s = 0
		for i in range(len(X)):
			s = s + (X[i][0] * W[0] + B[0] + X[i][1] * W[1] + B[1] - Y[i])*X[i][0]
		gradW11 = s / n

		s = 0
		for i in range(len(X)):
			s = s + (X[i][0] * W[0] + B[0] + X[i][1] * W[1] + B[1] - Y[i])*X[i][1]
		gradW12 = s / n

		s = 0
		for i in range(len(X)):
			s = s + (X[i][0] * W[0] + B[0] + X[i][1] * W[1] + B[1] - Y[i])
		gradB11 = s / n

		gradB12 = gradB11

		# gradient descent

		W[0] = W[0] - a*gradW11
		W[1] = W[1] - a*gradW12
		B[0] = B[0] - a*gradB11
		B[1] = B[1] - a*gradB12

	return [W, B]

def infer(x, model):

	return x[0] * model[0][0] + model[1][0] + x[1] * model[0][1] + model[1][1]


def print_results(x, y, X, Y, model):

	print(f'Weights: {model[0]}')
	print(f'Biases:  {model[1]}')

	print(f'Loss: {calculate_loss(X, Y, model)}')

	print(f'For inputs {x[0]} and {x[1]}, the output is {y}.')

def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = [50,100]
	prediction = infer(data, model)

	print_results(data, prediction, X, Y, model)


if __name__ == '__main__':
	main()