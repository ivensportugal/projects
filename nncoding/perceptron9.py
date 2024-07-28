# architecture: n - 1
# feature: vectorization and numpy

import pandas as pd
import numpy as np

# parameters
filename = 'input.csv'

# hyperparameters
l = 0.01 # learning rate

W = []
b = 0.1

def read_data(filename):

    df = pd.read_csv(filename)

    X = df.x.to_numpy()
    Y = df.y.to_numpy()

    return X, Y

def activate(z):
    return z

def calculate_loss(X, Y):

    n = len(X) if len(X.shape) == 1 else len(X.shape)[1]

    S = X*W+b - Y
    loss = S.dot(S) / (2*n)

    return loss

def train(X, Y):

    n_input = 1 if len(X.shape) == 1 else len(X.shape)[0]
    n = len(X) if len(X.shape) == 1 else len(X.shape)[1]

    global W
    global b

    W = np.full(n_input,0.1)

    while(True):

        loss = calculate_loss(X, Y)
        print(f'loss: {loss}')
        # input()
        if loss < 0.0001: break

        grad_w = ((activate(X*W+b)-Y)*X).sum() / n
        grad_b = (activate(X*W+b)-Y).sum() / n

        W = W - l*grad_w
        b = b - l*grad_b

    return [W, b, loss]

def infer(x, model):

    return activate(x*W+b)

def print_results(x, y, model):

    print(f'Weights: {model[0]}')
    print(f'Biases: {model[1]}')
    print(f'Loss: {model[2]}')

    print(f'For input {x}, the predicuted output is {y}.')

    

def main():

    X, Y = read_data(filename)
    model = train(X, Y)

    data = 150
    prediction = infer(data, model)
    
    print_results(data, prediction, model)

if __name__ == '__main__':
    main()