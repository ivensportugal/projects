import pandas as pd

# parameters
filename = 'input.csv'

# hyperparameters
a = 0.01 # learning rate
W = 0
B = 0

def read_data(filename):

	df = pd.read_csv(filename)
	X = df['x']
	Y = df['y']

	return X, Y

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(X)

	s=0
	for i in range(n):
		t = activate(X[i]*W+B) - Y[i]
		s = s + t*t
	loss = s/n

	return loss

def train(X, Y):

	n = len(X)
	loss = 0

	global W
	global B

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.01: break

		s=0
		for i in range(n):
			s = s + (activate(X[i]*W+B) - Y[i])*X[i]
		grad_w = s / n

		s=0
		for i in range(n):
			s = s + (activate(X[i]*W+B) - Y[i])
		grad_b = s / n

		W = W - a*grad_w
		B = B - a*grad_b

	return [W, B, loss]

def infer(x, model):

	y = x*W + B
	return y

def print_results(x, y, model):

	print(f'Weights: {model[0]}')
	print(f'Biases: {model[1]}')

	print(f'Loss: {model[2]}')

	print(f'For input {x}, the prediction is {y}.')

def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()