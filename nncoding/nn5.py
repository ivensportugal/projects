# architecture: 1 - 2 - 2

import pandas as pd

# parameters
filename = 'input3.csv'

# hyperparameters
l = 0.001 # learning rate

wl1 = 0.1
bl1 = 0.1
wl2 = 0.1
bl2 = 0.1

wL11 = wL12 = wL21 = wL22 = 0.1
bL1 = bL2 = 0.1

def read_data(filename):

	df = pd.read_csv(filename)

	X = df['x']
	Y1 = df['y1']
	Y2 = df['y2']
	Y = [Y1, Y2]

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

		zL1 = al1*wL11 + al2*wL12 + bL1
		zL2 = al1*wL21 + al2*wL22 + bL2

		aL1 = activate(zL1)
		aL2 = activate(zL2)

		y1 = aL1
		y2 = aL2

		t1 = y1-Y[0][i]
		t2 = y2-Y[1][i]

		s = s + t1*t1 + t2*t2

	loss = s/(2*n)

	return loss


def train(X, Y):

	n = len(X)

	global wl1, wl2, bl1, bl2
	global wL11, wL12, wL21, wL22, bL1, bL2

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.01: break

		swL11 = swL12 = swL21 = swL22 = 0
		sbL1 = sbL2 = 0

		swl1 = swl2 = 0
		sbl1 = sbl2 = 0

		for i in range(n):

			zl1 = X[i]*wl1 + bl1
			zl2 = X[i]*wl2 + bl2

			al1 = activate(zl1)
			al2 = activate(zl2)

			zL1 = al1*wL11 + al2*wL12 + bL1
			zL2 = al1*wL21 + al2*wL22 + bL2

			aL1 = activate(zL1)
			aL2 = activate(zL2)

			y1 = aL1
			y2 = aL2


			swL11 = swL11 + (y1-Y[0][i]) * al1
			swL12 = swL12 + (y1-Y[0][i]) * al2

			swL21 = swL21 + (y2-Y[1][i]) * al1
			swL22 = swL22 + (y2-Y[1][i]) * al2

			sbL1 = sbL1 + (y1-Y[0][i])
			sbL2 = sbL2 + (y2-Y[1][i])

			swl1 = (y1-Y[0][i]) * wL11 * X[i] + (y2-Y[1][i]) * wL21 * X[i]
			swl2 = (y1-Y[0][i]) * wL12 * X[i] + (y2-Y[1][i]) * wL22 * X[i]

		gradwL11 = swL11 / n
		gradwL12 = swL12 / n
		gradwL21 = swL21 / n
		gradwL22 = swL22 / n

		gradbL1 = sbL1 / n
		gradbL2 = sbL2 / n

		gradwl1 = swl1 / n
		gradwl2 = swl2 / n

		gradbl1 = sbl1 / n
		gradbl2 = sbl2 / n


		wL11 = wL11 - l * gradwL11
		wL12 = wL12 - l * gradwL12
		wL21 = wL21 - l * gradwL21
		wL22 = wL22 - l * gradwL22

		bL1 = bL1 - l * gradbL1
		bL2 = bL2 - l * gradbL2

		wl1 = wl1 - l * gradwl1
		wl2 = wl2 - l * gradwl2

		bl1 = bl1 - l * gradbl1
		bl2 = bl2 - l * gradbl2

	return [[[wl1,bl1],[wl2,bl2]],[[wL11,wL12,bL1],[wL21,wL22,bL2]],loss]

def infer(x, model):

	s = 0

	zl1 = x * model[0][0][0] + model[0][0][1]
	zl2 = x * model[0][1][0] + model[0][1][1]

	al1 = activate(zl1)
	al2 = activate(zl2)

	zL1 = al1 * model[1][0][0] + al2 * model[1][0][1] + model[1][0][2]
	zL2 = al1 * model[1][1][0] + al2 * model[1][1][1] + model[1][1][2]

	aL1 = activate(zL1)
	aL2 = activate(zL2)

	y1 = aL1
	y2 = aL2

	return [y1, y2]


def print_results(x, y, model):

	print(f'Hidden Layer: {model[0]}')
	print(f'Output Layer: {model[1]}')

	print(f'Loss: {model[2]}')

	print(f'For an input {x}, the predicted output is {y}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()