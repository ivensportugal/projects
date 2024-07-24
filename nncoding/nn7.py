# architecture: 2 - 2 - 1

import pandas as pd

# parameters
filename = 'input5.csv'

# hyperparameters
l = 0.01 # learning rate

wl11 = 0.1
wl12 = 0.1
wl21 = 0.1
wl22 = 0.1

bl1 = 0.1
bl2 = 0.1

wL1 = 0.1
wL2 = 0.1
bL = 0.1

def read_data(filename):

	df = pd.read_csv(filename)

	X1 = df['x1']
	X2 = df['x2']
	X = [X1, X2]
	Y = df['y']

	return X, Y

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(Y)

	s = 0
	for i in range(n):

		zl1 = X[0][i] * wl11 + X[1][i] * wl12 + bl1
		al1 = activate(zl1)

		zl2 = X[0][i] * wl21 + X[1][i] * wl22 + bl2
		al2 = activate(zl2)

		zL = al1 * wL1 + al2 * wL2 + bL
		aL = activate(zL)

		y = aL

		t = y - Y[i]
		s = s + t*t

	loss = s/(2*n)

	return loss


def train(X, Y):

	n = len(Y)

	global wl11, wl12, wl21, wl22
	global bl1, bl2

	global wL1, wL2
	global bL

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.037: break

		swl11 = swl12 = swl21 = swl22 = 0
		sbl1 = sbl2 = 0

		swL1 = swL2 = 0
		sbL = 0

		for i in range(n):

			zl1 = X[0][i] * wl11 + X[1][i] * wl12 + bl1
			al1 = activate(zl1)

			zl2 = X[0][i] * wl21 + X[1][i] * wl22 + bl2
			al2 = activate(zl2)

			zL = al1 * wL1 + al2 * wL2 + bL
			aL = activate(zL)

			y = aL

			swL1 = (y-Y[i]) * al1
			swL2 = (y-Y[i]) * al2
			sbL = (y-Y[i])

			swl11 = (y-Y[i]) * wL1 * X[0][i]
			swl12 = (y-Y[i]) * wL1 * X[1][i]
			swl21 = (y-Y[i]) * wL2 * X[0][i]
			swl22 = (y-Y[i]) * wL2 * X[1][i]
			sbl1 = (y-Y[i]) * wL1
			sbl2 = (y-Y[i]) * wL2

		grad_wL1 = swL1 / n
		grad_wL2 = swL2 / n
		grad_bL = sbL / n

		grad_wl11 = swl11 / n
		grad_wl12 = swl12 / n
		grad_wl21 = swl21 / n
		grad_wl22 = swl22 / n
		grad_bl1 = sbl1 / n
		grad_bl2 = sbl2 / n

		wL1 = wL1 - l * grad_wL1
		wL2 = wL2 - l * grad_wL2
		bL = bL - l * grad_bL

		wl11 = wl11 - l * grad_wl11
		wl12 = wl12 - l * grad_wl12
		wl21 = wl21 - l * grad_wl21
		wl22 = wl22 - l * grad_wl22
		bl1 = bl1 - l * grad_bl1
		bl2 = bl2 - l * grad_bl2

	return [[[wl11,wl12,bl1],[wl21,wl22,bl2]],[wL1,wL2,bL],loss]

def infer(x, model):

	zl1 = x[0] * model[0][0][0] + x[1] * model[0][0][1] + model[0][0][2]
	zl2 = x[0] * model[0][1][0] + x[1] * model[0][1][1] + model[0][1][2]

	al1 = activate(zl1)
	al2 = activate(zl2)

	zL = al1 * model[1][0] + al2 * model[1][1] + model[1][2]
	aL = activate(zL)

	y = aL

	return y

def print_results(x, y, model):

	print(f'Hidden Layer: {model[0]}')
	print(f'Output Layer: {model[1]}')

	print(f'Loss: {model[2]}')

	print(f'For an input {x}, the predicted output is {y}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = [50,51]
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()