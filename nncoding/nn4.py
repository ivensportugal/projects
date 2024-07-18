import pandas as pd

# architecture: 1-2-1

# parameters
filename = 'input.csv'

# hyperparameters
l = 0.001 # learning rate

wl1 = 0.1
wl2 = 0.5

bl1 = 0.1
bl2 = 0.1

wL1 = 0.1
wL2 = 0.1

bL = 0.1

def read_data(filename):

	df = pd.read_csv(filename)

	X = df['x']
	Y = df['y']

	return X, Y

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(X)

	s = 0
	for i in range(n):
		
		zl1 = X[i]*wl1 + bl1
		zl2 = X[i]*wl2 + bl2

		al1 = activate(zl1)
		al2 = activate(zl2)

		zL = al1*wL1 + al2*wL2 + bL

		aL = activate(zL)

		y = aL

		t = y-Y[i]
		s = s + t*t

	loss = s / (2*n)

	return loss

def train(X, Y):

	n = len(X)

	global wl1
	global wl2

	global bl1
	global bl2

	global wL1
	global wL2

	global bL

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.001: break

		swL1 = 0
		swL2 = 0
		sbL = 0

		swl1 = 0
		swl2 = 0
		sbl1 = 0
		sbl2 = 0
		for i in range(n):

			zl1 = X[i]*wl1 + bl1
			zl2 = X[i]*wl2 + bl2

			al1 = activate(zl1)
			al2 = activate(zl2)

			zL = al1*wL1 + al2*wL2 + bL

			aL = activate(zL)

			y = aL

			swL1 = swL1 + (y-Y[i]) * al1
			swL2 = swL2 + (y-Y[i]) * al2
			sbL = sbL + (y - Y[i])

			swl1 = swl1 + (y-Y[i]) * wL1 * X[i]
			swl2 = swl2 + (y-Y[i]) * wL2 * X[i]
			sbl1 = sbl1 + (y-Y[i]) * wL1
			sbl2 = sbl2 + (y-Y[i]) * wL2

		grad_wL1 = swL1/n
		grad_wL2 = swL2/n
		grad_bL = sbL/n

		grad_wl1 = swl1/n
		grad_wl2 = swl2/n
		grad_bl1 = sbl1/n
		grad_bl2 = sbl2/n

		wL1 = wL1 - l*grad_wL1
		wL2 = wL2 - l*grad_wL2
		bL = bL - l*grad_bL

		wl1 = wl1 - l*grad_wl1
		wl2 = wl2 - l*grad_wl2
		bl1 = bl1 - l*grad_bl1
		bl2 = bl2 - l*grad_bl2

	return [[[wl1,bl1],[wl2,bl2]],[[wL1,wL2],bL], loss]


def infer(x, model):

	zl1 = x*wl1 + bl1
	zl2 = x*wl2 + bl2

	al1 = activate(zl1)
	al2 = activate(zl2)

	zL = al1*wL1 + al2*wL2 + bL

	aL = activate(zL)

	y = aL

	return y


def print_results(x, y, model):

	print(f'hidden layer: {model[0][0]} and {model[0][1]}')
	print(f'output layer: {model[1]}')

	print(f'loss: {model[2]}')

	print(f'For an input {x}, the predicted output is {y}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()