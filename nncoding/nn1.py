import pandas as pd

# architecture: 1-1-1

# parameters
filename = 'input.csv'

# hyperparameters
l = 0.01 # learning rate

w = 0.1
b = 0.1
wL = 0.1
bL = 0.1

def read_data(filename):

	X = []
	Y = []

	df = pd.read_csv(filename)
	n = len(df)

	X = df['x']
	Y = df['y']

	return X, Y

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(X)

	s = 0
	for i in range(n):
		t = activate(activate(X[i]*w+b) * wL + bL) - Y[i]
		s = s + t*t
	loss = s / (2*n)

	return loss

def train(X, Y):

	n = len(X)
	loss = 0

	while(True):

		global w
		global b
		global wL
		global bL

		loss = calculate_loss(X, Y)
		if loss < 0.01: break

		# calculating gradients
		s = 0
		for i in range(n):
			s = s + ((activate(activate(X[i]*w+b)*wL+bL)) - Y[i]) * activate(X[i]*w+b)
		grad_wL = s/n

		s = 0
		for i in range(n):
			s = s + ((activate(activate(X[i]*w+b)*wL+bL)) - Y[i])
		grad_bL = s/n

		s = 0
		for i in range(n):
			s = s + ((activate(activate(X[i]*w+b)*wL+bL)) - Y[i]) * wL * activate(X[i])
		grad_w = s/n

		s = 0
		for i in range(n):
			s = s + ((activate(activate(X[i]*w+b)*wL+bL)) - Y[i]) * wL
		grad_b = s/n

		wL = wL - l*grad_wL
		bL = bL - l*grad_bL
		w  = w  - l*grad_w
		b  = b  - l*grad_b

	return [[w,b],[wL,bL], loss]

def infer(x, model):

	y = activate(activate(x*model[0][0]+model[0][1]) * model[1][0] + model[1][1])
	return y

def print_results(x, y, model):

	print(f'Hidden Layer:')
	print(f': {model[0]}')
	print(f'Output Layer:')
	print(f': {model[1]}')

	print(f'Loss: {model[2]}')

	print(f'For an input {x}, the prediction is {y}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)

	print_results(data, prediction, model)


if __name__ == '__main__':
	main()