# architecture: 2 - 1 - 2

import pandas as pd

# parameters
filename = 'input4.csv'

# hyperparameters
l = 0.00006 # learning rate

wl11 = 0.5
wl12 = 0.5
bl1 = 0.5

wL11 = 0.5
bL1 = 0.5

wL21 = 0.5
bL2 = 0.5

def read_data(filename):

	df = pd.read_csv(filename)
	
	X1 = df['x1']
	X2 = df['x2']
	X = [X1, X2]

	Y1 = df['y1']
	Y2 = df['y2']
	Y = [Y1, Y2]

	return [X, Y]

def activate(z):
	return z

def calculate_loss(X, Y):

	n = len(X[0])

	s = 0
	for i in range(n):
		
		zl1 = X[0][i] * wl11 + X[1][i] * wl12 + bl1
		al1 = activate(zl1)

		zL11 = al1 * wL11 + bL1
		zL21 = al1 * wL21 + bL2

		aL11 = activate(zL11)
		aL21 = activate(zL21)

		y1 = aL11
		y2 = aL21

		t1 = (y1 - Y[0][i])
		t2 = (y2 - Y[1][i])

		s = s + t1*t1 + t2*t2

	loss = s/(2*n)

	return loss


def train(X, Y):

	n = len(X[0])

	global wl11, wl12, bl1
	global wL11, wL21, bL1, bL2

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 7: break

		swL11 = 0
		swL21 = 0
		sbL1 = 0
		sbL2 = 0
		swl11 = 0
		swl12 = 0
		sbl1 = 0

		for i in range(n):

			zl1 = X[0][i] * wl11 + X[1][i] * wl12 + bl1
			al1 = activate(zl1)

			zL11 = al1 * wL11 + bL1
			zL21 = al1 * wL21 + bL2

			aL11 = activate(zL11)
			aL21 = activate(zL21)

			y1 = aL11
			y2 = aL21

			swL11 = swL11 + (y1 - Y[0][i]) * al1
			swL21 = swL21 + (y2 - Y[1][i]) * al1
			sbL1 = sbL1 + (y1 - Y[0][i])
			sbL2 = sbL2 + (y2 - Y[1][i])

			swl11 = swl11 + (y1 - Y[0][i]) * wL11 * X[0][i] + (y2 - Y[1][i]) * wL21 * X[0][i]
			swl12 = swl12 + (y1 - Y[0][i]) * wL11 * X[1][i] + (y2 - Y[1][i]) * wL21 * X[1][i]

			sbl1 = sbl1 + (y1 - Y[0][i]) * wL11 + (y2 - Y[1][i]) * wL21

		grad_swL11 = swL11 / n
		grad_swL21 = swL21 / n
		grad_sbL1 = sbL1 / n
		grad_sbL2 = sbL2 / n

		grad_swl11 = swl11 / n
		grad_swl12 = swl12 / n
		grad_sbl1 = sbl1 / n


		wL11 = wL11 - l * grad_swL11
		wL21 = wL21 - l * grad_swL21
		bL1 = bL1 - l * grad_sbL1
		bL2 = bL2 - l * grad_sbL2

		wl11 = wl11 - l * grad_swl11
		wl12 = wl12 - l * grad_swl12
		bl1 = bl1 - l * grad_sbl1

	return [[wl11, wl12, bl1], [[wL11, bL1],[wL21, bL2]], loss]


def infer(x, model):

	zl1 = x[0] * model[0][0] + x[1] * model[0][1] + model[0][2]
	al1 = activate(zl1)

	zL11 = al1 * model[1][0][0] + model[1][0][1]
	zL21 = al1 * model[1][1][0] + model[1][1][1]

	aL11 = activate(zL11)
	aL21 = activate(zL21)

	y1 = aL11
	y2 = aL21

	return [y1, y2]

def print_results(x, y, model):

	print(f'Hidden Layer: {model[0]}')
	print(f'Output Layer: {model[1]}')

	print(f'Loss: {model[2]}')

	print(f'For input {x}, the predicted outputs are {y[0]} and {y[1]}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = [51, 50]
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()