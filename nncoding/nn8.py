# architecture: n - n - 1
# feature: vectorization in neural network

import pandas as pd
import numpy as np

# parameters
filename = 'input2.csv'

# hyperparameters
l = 0.001 # learning rate

Wl = []
Bl = []

WL = []
bL = []

def read_data(filename):

    df = pd.read_csv(filename)

    x_list = []
    y_list = []

    for col in df.columns:
        if 'x' in col:
            x_list.append(col)
        else:
            y_list.append(col)

    X = df[x_list].to_numpy()
    Y = df[y_list].to_numpy()

    return X, Y

def activate(Z):
    return np.apply_along_axis(lambda z: z, 0, Z)

def calculate_loss(X, Y):

    Zl = Wl @ np.linalg.matrix_transpose(X) + Bl
    Al = activate(Zl)

    zL = WL @ Al + bL
    aL = activate(zL)

    y = aL

    t = np.matrix_transpose(y) - Y
    loss = (np.linalg.matrix_transpose(t)@t) / (2*X.shape[0])

    return loss[0][0]


def train(X, Y):

    n_input = X.shape[1]
    n = X.shape[0]

    global Wl
    global Bl
    global WL
    global bL

    Wl = np.full((n_input, n_input),0.1)
    Bl = np.full((n_input,1),0.1)

    WL = np.full((1,n_input), 0.1)
    bL = np.full((1,1), 0.1)

    while(True):

        loss = calculate_loss(X, Y)
        if loss < 0.001: break

        Zl = Wl @ np.linalg.matrix_transpose(X) + Bl
        Al = activate(Zl)

        zL = WL @ Al + bL
        aL = activate(zL)

        y = aL

        grad_WL = np.linalg.matrix_transpose(Al @ (np.linalg.matrix_transpose(y) - Y)) / n
        grad_bL = np.linalg.matrix_transpose(np.ones((1,n)) @ (np.linalg.matrix_transpose(y) - Y)) / n

        grad_Wl = (np.linalg.matrix_transpose((np.linalg.matrix_transpose(y)-Y)@WL) @ X)/n
        grad_bl = ((np.linalg.matrix_transpose((np.linalg.matrix_transpose(y)-Y)@WL)) @ np.ones((n,1))) / n

        WL = WL - l*grad_WL
        bL = bL - l*grad_bL

        Wl = Wl - l*grad_Wl
        Bl = Bl - l*grad_bl

    return [[Wl,Bl],[WL,bL],loss]

def infer(X, model):
    
    Zl = model[0][0] @ np.linalg.matrix_transpose(X) + model[0][1]
    Al = activate(Zl)

    zL = model[1][0] @ Al + model[1][1]
    aL = activate(zL)

    y = aL

    return y

def print_results(x, y, model):

    print(f'Hidden Layer: {model[0]}')
    print(f'Output Layer: {model[1]}')
    print(f'Loss: {model[2]}')

    print(f'For input {x}, the predicted output is {y}.')

def main():

    X, Y = read_data(filename)
    model = train(X, Y)

    data = [[50, 51]]
    prediction = infer(data, model)

    print_results(data, prediction, model)

if __name__ == '__main__':
    main()