# architecture: recurrent neural network
# feature: none

import pandas as pd
import numpy as np

# parameters
filename = 'inputrnn1.csv'

# hyperparameters
l = 0.01 # learning rate

why = 0.1
whh = 0.1
whx = 0.1

by = 0.1
bh = 0.1

def read_data(filename):

	df = pd.read_csv(filename)

	Xcol = []
	Ycol = []
	for col in df.columns:
		if 'x' in col: Xcol.append(col)
		else: Ycol.append(col)

	X = df[Xcol].to_numpy()
	Y = df[Ycol].to_numpy()

	return X, Y

def activate_sigma(x):
	return max(0,x) # ReLU

def activate_Sigma(X):
	return np.array([activate_sigma(x) for x in X])

def grad_sigma(x):
	return np.where(x>0, 1, 0) #ReLU

def grad_Sigma(X):
	return np.array([grad_sigma(x) for x in X])

def activate_phi(x):
	return x # identity

def activate_Phi(X):
	return np.array([activate_phi(x) for x in X])

def grad_phi(x):
	return 1 # identity

def grad_Phi(X):
	return np.array([grad_phi(x) for x in X])
	

def calculate_loss(X, Y):

	n = X.shape[0]
	m = X.shape[1]

	s = 0
	for i in range(n):
		y = infer(X[i])
		t = y - Y[i]
		s = s + t @ t
	loss = s / (2*m*n)

	return loss


def train(X, Y):

	n = X.shape[0]
	m = X.shape[1] # or k

	global l
	global why, whx, whh
	global by, bh

	while(True):

		loss = calculate_loss(X, Y)
		print(f'loss: {loss}')
		if loss < 0.1: break

		s_why = 0
		s_by  = 0
		s_whx = 0
		s_whh = 0
		s_bh  = 0
		for i in range(n):

			Hprevious = 0
			Hi = np.array([Hprevious := activate_sigma(whx * X_ + whh * Hprevious + bh) for X_ in X[i]])
			# if i==0:
			# 	print(f'###whx: {whx}')
			# 	print(f'###why: {why}')
			# 	print(f'###bh: {bh}')
			# 	print(f'####Hi: {Hi}')
			# 	input()

			Yi = activate_Phi(why * Hi + by)
			grad_Yi = (Yi - Y[i]) / (n)

			gradYi_Hi = why * activate_Phi(why * Hi + by)

			X_ = np.append(X[i][1:], 0)
			# print(f'whx: {whx}')
			# print(f'X_: {X_}')
			# print(f'whh: {whh}')
			# print(f'Hi: {Hi}')
			# print(f'bh: {bh}')
			gradHi_Hi = whh * grad_Sigma(whx * X_ + whh * Hi + bh)

			grad_Hinext = 0
			grad_Hi = np.array([grad_Hinext := gYi * gYiHi + grad_Hinext * gHiHi for gYi, gYiHi, gHiHi in zip(grad_Yi, gradYi_Hi, gradHi_Hi)]) / (n)

			gradYi_why = Hi * grad_Phi(why * Hi + by)

			s_why = s_why + grad_Yi @ gradYi_why
			s_by  = s_by  + grad_Yi @ np.ones(m)

			Hi_ = np.insert(Hi[:-1], 0, 0)
			gradHi_whx = X[i] * grad_Sigma(whx * X[i] + whh * Hi_ + bh)
			# print(f'X[i]: {X[i]}')
			# print(f'Hi_: {Hi_}')
			gradHi_whh = Hi_  * grad_Sigma(whx * X[i] + whh * Hi_ + bh)

			# print(gradYi_why)
			# print(gradHi_whx)

			s_whx = s_whx + grad_Hi @ gradHi_whx
			s_whh = s_whh + grad_Hi @ gradHi_whh
			s_bh  = s_bh  + grad_Hi @ np.ones(m)

			# if s_whh == 0 and i!=0:
			# 	print(f'grad_Hi: {grad_Hi}')
			# 	print(f'gradHi_whh: {gradHi_whh}')

		print(f's_why: {s_why}')
		print(f's_by: {s_by}')
		print(f's_whx: {s_whx}')
		print(f's_whh: {s_whh}')
		print(f's_bh: {s_bh}')

		grad_why = s_why / n
		grad_by  = s_by / n
		grad_whx = s_whx / n
		grad_whh = s_whh / n
		grad_bh  = s_bh / n

		print(f'grad_why: {grad_why}')
		print(f'grad_by: {grad_by}')
		print(f'grad_whx: {grad_whx}')
		print(f'grad_whh: {grad_whh}')
		print(f'grad_bh: {grad_bh}')

		why = why - l * grad_why
		by = by - l * grad_by
		whx = whx - l * grad_whx
		whh = whh - l * grad_whh
		bh = bh - l * grad_bh

		print(f'why: {why}')
		print(f'by: {by}')
		print(f'whx: {whx}')
		print(f'whh: {whh}')
		print(f'bh: {bh}')

		print('')

		# if s_whh == 0: input()

	return [why, by, whx, whh, bh, loss]



def infer(X):

	global whx, whh, why
	global bh, by

	ht_previous = 0
	ht = np.array([ht_previous := activate_sigma(whx * x + whh * ht_previous + bh) for x in X])
	yt = activate_Phi(why * ht + by)

	return yt


def print_results(x, y, model):

	print(f'input {x}')
	print(f'output: {y}')
	print(f'why: {model[0]}')
	print(f'by: {model[1]}')
	print(f'whx: {model[2]}')
	print(f'whh: {model[3]}')
	print(f'bh: {model[4]}')
	print(f'loss: {model[-1]}')


def main():

	X, Y = read_data(filename)
	model = train(X, Y)

	data = np.array([1,5,10,15])
	prediction = infer(data)

	print_results(data, prediction, model)

if __name__ == '__main__':
	main()