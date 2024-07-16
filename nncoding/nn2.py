import pandas as pd

# architecture: 2-1-1

# parameters
filename = 'input2.csv'

#hyperparameters
l = 0.001 # learning rate
wl1 = 0.1
wl2 = 0.1
bl = 0.1
wL = 0.1
bL = 0.1

def read_data(filename):

	df = pd.read_csv(filename)
	X1 = df['x1']
	X2 = df['x2']
	X = [X1,X2]
	Y = df['y']

	return X, Y

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(Y)

	s = 0
	for i in range(n):
		zl = X[0][i] * wl1 + X[1][i] * wl2 + bl
		al = activate(zl)
		zL = al * wL + bL
		aL = activate(zL)
		t = aL - Y[i]
		s = s + t*t
	loss = s / (2*n)

	return loss

def train(X, Y):

	n = len(Y)

	global wl1
	global wl2
	global bl
	global wL
	global bL

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 00.1: break

		s = 0
		for i in range(n):
			s = s + (activate((activate(X[0][i]*wl1+X[1][i]*wl2+bl))*wL+bL) - Y[i]) * activate(X[0][i]*wl1+X[1][i]*wl2+bl)
		grad_wL = s/n

		s = 0
		for i in range(n):
			s = s + (activate((activate(X[0][i]*wl1+X[1][i]*wl2+bl))*wL+bL) - Y[i])
		grad_bL = s/n

		s = 0
		for i in range(n):
			s = s + (activate((activate(X[0][i]*wl1+X[1][i]*wl2+bl))*wL+bL) - Y[i]) * wL * X[0][i]
		grad_wl1 = s/n

		s = 0
		for i in range(n):
			s = s + (activate((activate(X[0][i]*wl1+X[1][i]*wl2+bl))*wL+bL) - Y[i]) * wL * X[1][i]
		grad_wl2 = s/n

		s = 0
		for i in range(n):
			s = s + (activate((activate(X[0][i]*wl1+X[1][i]*wl2+bl))*wL+bL) - Y[i])
		grad_bl = s/n

		wL = wL - l * grad_wL
		bL = bL - l * grad_bL
		wl1 = wl1 - l * grad_wl1
		wl2 = wl2 - l * grad_wl2
		bl = bl - l * grad_bl

	return [[wl1, wl2, bl], [wL, bL], loss]

def infer(x, model):

	zl = x[0]*wl1 + x[1]*wl2 + bl
	al = activate(zl)
	zL = al*wL + bL
	aL = activate(zL)

	y = aL

	return y

def print_results(x, y, model):

	print(f'Hidden layer: {model[0]}')
	print(f'Output layer: {model[1]}')

	print(f'Loss: {model[2]}')

	print(f'For an input {x}, the prediction is {y}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = [50,5]
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()