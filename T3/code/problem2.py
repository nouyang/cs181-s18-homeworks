# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples
        self.weights = [] #of len(x)
        self.b = 0.0

    # Implement this!
    def fit(self, X, Y):
        # as per https://www.cs.cmu.edu/~avrim/ML10/lect0125.pdf
        self.X = X
        self.Y = Y
        X = np.array(X)
        self.weights = np.zeros((1, X.shape[1]))
        self.b = 0.0
        # normalize X
        row_sums = X.sum(axis=1) 
        X = X / row_sums[:, np.newaxis]
        #rand_idx = np.random.rand(len() # how many random numbers valued b/tw 0 and 1
        print(len(X))
        print(len(Y))
        rand_idx = np.random.permutation(self.numsamples)
        for i in rand_idx:
            print(X[i])
            y_hat = np.dot(self.weights, X[i])
            print(y_hat)
            print(Y[i])
            if Y[i] > y_hat: # 1, -1
                self.weights += X[i] 
            if Y[i] < y_hat: # -1, 1
                self.weights -= X[i]
            # use sparse natrices
            # e.g. matrix = {(0, 3): 1, (2, 1): 2, (4, 3): 3}
            # print(matrix.get((1, 3), 0)) # return 0 if not in dictionary


        # Implement this!
    def predict(self, X):
        return np.ones(len(X))
        # return array of 1's and -1's for classifying

# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = Nh
        self.numsamples = numsamples

        # Implement this!
    def fit(self, X, Y):
        pass

    # Implement this!
    # def predict(self, X):



# Do not change these three lines.
#data = np.loadtxt("data.csv", delimiter=',')
data = np.loadtxt("data_short.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
#numsamples = 20000
numsamples = 100

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)
