import pandas as pd

# parameters
filename = 'input2.csv'

# hyperparameters
a = 0.001 # learning rate
W = [0,0]
B = 0

def read_data(filename):

	df = pd.read_csv(filename)
	X = df[['x1','x2']].values.tolist()
	Y = df['y'].tolist()

	return X, Y

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(X)

	global B

	s = 0
	for i in range(n):
		t = activate(X[i][0]*W[0] + X[i][1]*W[1] + B) - Y[i]
		s = s + t*t
	loss = s / (2*n)

	return loss

def train(X, Y):

	n = len(X)

	global B

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.01: break

		s0=0
		s1=0
		for i in range(n):
			s0 = s0 + (activate(X[i][0]*W[0] + X[i][1]*W[1] + B) - Y[i])*X[i][0]
			s1 = s1 + (activate(X[i][0]*W[0] + X[i][1]*W[1] + B) - Y[i])*X[i][1]
		grad_w0 = s0/n
		grad_w1 = s1/n

		s=0
		for i in range(n):
			s = s + activate(X[i][0]*W[0] + X[i][1]*W[1] + B) - Y[i]
		grad_b = s/n

		W[0] = W[0] - a*grad_w0
		W[1] = W[1] - a*grad_w1
		B = B - a*grad_b

	return [W, B, loss]

def infer(X, model):
	y = activate(X[0]*model[0][0] + X[1]*model[0][1] + model[1])
	return y


def print_results(x, y, model):

	print(f'Weights: {model[0]}')
	print(f'Biases: {model[1]}')

	print(f'Losses: {model[2]}')

	print(f'For an input {x}, the predicted value is {y}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = [50,51]
	prediction = infer(data, model)

	print_results(data, prediction, model)


if __name__ == '__main__':
	main()