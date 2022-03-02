import pandas as pd
import numpy as np

train_df = pd.read_csv("HeartTrain.csv", index_col=0)
test_df = pd.read_csv("HeartTest.csv", index_col=0)

x = train_df.drop("labels", axis=1)
y = train_df["labels"]

def perceptron(x, y, b, w_init, eta, epoch):
    """
    Implements a binary single-example perceptron with margin
    Inputs:    x a feature matrix containing an example on each row [pandas DataFrame of shape n x d]
               y a vector with the class (either 1 or 0) of each example  [list or numpy array of size n]
               b a margin [int]
               w_init a vector with the initial weight values (intercept in w_init[0]) [list or numpy array of size d+1]
               eta a fixed learning rate [int]
               epoch the maximal number of iterations (1 epoch = 1 iteration
                       of the "repeat" loop in the lecture slides) [int]
    Output:    A weight vector [list or numpy array of size d+1] (intercept in w[0])
    """


    w = w_init
    x = x.to_numpy()

    for k in range(epoch):

        i = (k % x.shape[0]) + 1
        xi = x[i]
        yi = y[i]
        #print(w.T @ xk)
        if y[i] == 0:
            if(w.T @ xi) > 0 and (w.T @ xi) <= b:
                print("Classified correctly with too small margin")
                w = w - (eta * k) @ xi
                print((eta * k))
                print((eta * k) *xi)

        if y[i] == 1:
            print(w[1:].T @ xi)
            if(w[1:].T @ xi) < 0 and (w[1:].T @ xi) <= b:
                w = w - (eta * k) @ xi



    return w


p = perceptron(x, y.to_numpy(), 2, w_init=np.concatenate(1, np.zeros([x.shape[1]+ 1, 1])), eta=0.1, epoch=10)
print(p.shape)
print(p)
