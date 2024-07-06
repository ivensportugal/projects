import pandas as pd
import csv

# hyperparameters
a = 0.01 # learning rate
w = 0 # weight
b = 0 # b

# read data

X = []
Y = []

with open('input.csv', mode='r') as file:
	csvFile = csv.reader(file)
	for line in csvFile:
		X.append(int(line[0]))
		Y.append(int(line[1]))

# actual learning
while(True):
	gradient_w = 0
	gradient_b = 0
	for i in range(len(X)):

		x = X[i]
		y = Y[i]

		gradient_w = gradient_w + (x*w-y)*x
		gradient_b = gradient_s + (x*w-y)

	w = w - (a*gradient_w)/len(X)
	b = b - (a*gradient_b)/len(X)

	# Loss

	s = 0
	for i in range(len(X)):

		x = X[i]
		y = Y[i]

		s = (x*w - y) * (x*w - y)

	L = s/(2*len(X))

	if (L < 0.01): break

print(f"The value of w is: {w}")
print(f"The value of b is: {b}")
x = 800
y = x*w+b
print(f"For an input {x}, the inference is {y}")