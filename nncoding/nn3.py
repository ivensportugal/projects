import pandas as pd

# architecture: 1-1-2

# parameters
filename = 'input3.csv'

# hyperparameters
l = 0.001 # learning rate
wl = 0.1
bl = 0.1
wL1 = 0.1
wL2 = 0.1
bL1 = 0.1
bL2 = 0.1

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

	s1 = 0
	s2 = 0

	for i in range(n):

		zl = X[i]*wl+bl
		al = activate(zl)

		zL1 = al*wL1+bL1
		aL1 = activate(zL1)
		y1 = aL1

		zL2 = al*wL2+bL2
		aL2 = activate(zL2)
		y2 = aL2

		t1 = y1-Y[0][i]
		t2 = y2-Y[1][i]

		s1 = s1 + t1*t1
		s2 = s2 + t2*t2

	loss = (s1 + s2) / (2*n)

	return loss


def train(X, Y):

	n = len(X)

	global wl
	global bl
	global wL1
	global wL2
	global bL1
	global bL2

	while(True):

		loss = calculate_loss(X, Y)
		if loss < 0.01: break

		swL1 = 0
		swL2 = 0
		sbL1 = 0
		sbL2 = 0
		swl = 0
		sbl = 0
		for i in range(n):
			zl = X[i]*wl+bl
			al = activate(zl)

			zL1 = al*wL1+bL1
			aL1 = activate(zL1)
			y1 = aL1

			zL2 = al*wL2+bL2
			aL2 = activate(zL2)
			y2 = aL2

			swL1 = swL1 + (y1 - Y[0][i])*al
			swL2 = swL2 + (y2 - Y[1][i])*al

			sbL1 = sbL1 + (y1 - Y[0][i])
			sbL2 = sbL2 + (y2 - Y[1][i])

			swl = swl + (y1 - Y[0][i])*wL1*X[i] + (y2 - Y[1][i])*wL2*X[i]
			sbl = sbl + (y1 - Y[0][i])*wL1      + (y2 - Y[1][i])*wL2

		grad_wL1 = swL1 / n
		grad_wL2 = swL2 / n

		grad_bL1 = sbL1 / n
		grad_bL2 = sbL2 / n

		grad_wl  = swl / n
		grad_bl  = sbl / n


		wL1 = wL1 - l*grad_wL1
		wL2 = wL2 - l*grad_wL2

		bL1 = bL1 - l*grad_bL1
		bL2 = bL2 - l*grad_bL2

		wl = wl - l*grad_wl
		bl = bl - l*grad_bl


	return [[wl,bl],[[wL1,bL1],[wL2,bL2]],loss]


def infer(x, model):

	zl = x*model[0][0] + model[0][1]
	al = activate(zl)

	zL1 = al*wL1 + bL1
	zL2 = al*wL2 + bL2

	aL1 = activate(zL1)
	aL2 = activate(zL2)

	y1 = aL1
	y2 = aL2

	return [y1,y2]

def print_results(x, y, model):

	print(f'Hidden Layer: {model[0]}')
	print(f'Output Layer: {model[1][0]} and {model[1][1]}')

	print(f'Loss: {model[2]}')

	print(f'For an input {x}, the predicted values are {y[0]} and {y[1]}.')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = 50
	prediction = infer(data, model)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()