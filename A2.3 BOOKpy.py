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
    #x.insert(0, "1", 1, True)
    x = x.to_numpy()

    for k in range(epoch):

        i = (k % x.shape[0])
        xi = x[i]

        #print(w[1:].T @ xi)
        if y[i] == 0:
            print(w.T @ xi)
            if(w.T @ xi) > 0 or (w.T @ xi) <= b:
                #print("Classified correctly with too small margin")
                update = np.multiply(eta * k, xi)
                update = update.reshape((update.shape[0], 1))
                w += update
                #print((eta * k))
                #print((eta * k)  * xi)

        if y[i] == 1:
            print(w.T @ xi)
            if(w.T @ xi) < 0 or (w.T @ xi) <= b:
                update = np.multiply(eta * k,  xi)
                update = update.reshape((update.shape[0], 1))
                w += update



    return w


w_init = np.zeros([x.shape[1], 1])

p = perceptron(x, y.to_numpy(), 2, w_init=w_init, eta=0.1, epoch=10)
print(p)
