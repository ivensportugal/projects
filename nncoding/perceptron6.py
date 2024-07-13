import pandas as pd

# parameters
filename = 'input3.csv'

# hyperparameters
a = 0.01 # learning rate
W = [0,0]
B = [0,0]

def read_data(filename):

	df = pd.read_csv(filename)
	X = df['x'].tolist()
	Y = df[['y1','y2']].values.tolist()

	return X, Y

def activate(z):
	return z

def calculate_loss0(X,Y):
	n = len(X)
	s = 0
	for i in range(n):
		t = activate(X[i]*W[0]+B[0] - Y[i][0])
		s = s+t*t
	loss = s/(2*n)
	return loss

def calculate_loss1(X,Y):
	n = len(X)
	s = 0
	for i in range(n):
		t = activate(X[i]*W[1]+B[1] - Y[i][1])
		s = s+t*t
	loss = s/(2*n)
	return loss

def train(X, Y):

	n = len(X)
	loss = 0

	while(True):

		loss0 = calculate_loss0(X, Y)
		loss1 = calculate_loss1(X, Y)
		if loss0 < 0.01 and loss1 < 0.01: break

		s0 = 0
		s1 = 0
		for i in range(n):
			s0 = s0 + (activate(X[i]*W[0]+B[0]) - Y[i][0])*X[i]
			s1 = s1 + (activate(X[i]*W[1]+B[1]) - Y[i][1])*X[i]
		grad_w0 = s0 / n
		grad_w1 = s1 / n

		s0 = 0
		s1 = 0
		for i in range(n):
			s0 = s0 + (activate(X[i]*W[0]+B[0]) - Y[i][0])
			s1 = s1 + (activate(X[i]*W[1]+B[1]) - Y[i][1])
		grad_b0 = s0/n
		grad_b1 = s1/n

		W[0] = W[0] - a*grad_w0
		W[1] = W[1] - a*grad_w1

		B[0] = B[0] - a*grad_b0
		B[1] = B[1] - a*grad_b1

	return [W,B,[loss0,loss1]]

def infer(x, model):

	y0 = activate(x*model[0][0]+model[1][0])
	y1 = activate(x*model[0][1]+model[1][1])

	return [y0,y1]

def print_results(x, y, model):

	print(f'Weights: {model[0]}')
	print(f'Biases: {model[1]}')

	print(f'Losses: {model[2]}')

	print(f'For input {x}, the prediction is {y}.')

def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()