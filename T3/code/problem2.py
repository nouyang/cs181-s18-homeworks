# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron
from numpy.core.umath_tests import inner1d


# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples
        self.alphas =  {} #of len(x)
        self.b = 0.0
        self.SV_indices = []

    # Implement this!
    def fit(self, X, Y):
        # as per https://www.cs.cmu.edu/~avrim/ML10/lect0125.pdf
        self.X = X
        print('asdfsadf', X.shape)
        self.Y = Y
        X = np.array(X)
        # K(x,x2 ) = x.T *  x2
        np.random.seed(314159)
        SV_indices = set()
        alphas = {}
        
        #rand_idx = np.random.permutation(self.numsamples)
         
        print(X.shape)
        for i in range(self.numsamples):
            t = np.random.randint(X.shape[0])
            #FIXED: not just permute through all training examples; instead pick ahead of time number of iters, and sample that many times
            xt = X[t]
            y_hat = np.sum(alphas[idx]*inner1d(xt,X[idx]) for idx in SV_indices)
            y_true = Y[t]
            if y_true * y_hat <= 0:
                SV_indices.add(t)
                alphas.update({t:y_true})
        self.alphas = alphas
        self.SV_indices = SV_indices

    def predict(self, X):
        print(X.shape)
        print("========= *** Predicting *** ==========")
        print('support vector indices', self.SV_indices)
        y_hats = []
        for xt in X:
            y_hat = np.sum( [self.alphas[sv_i] * np.dot(xt,X[sv_i]) \
                    for sv_i in self.SV_indices] )
            y_hats.append(y_hat)
        y_hats = np.array(y_hats)
        print(y_hats.shape)
        print(X.shape)
        print('y_hats', y_hats)
        print('alphas', self.alphas)
        return np.sign(y_hats)

        # X = np.array(X)
        # print(X.shape)
        # kern = inner1d(X, X)
        # # self.weights, and 0 if not in self.weights
        # n = X.shape[0] 
        # weights = np.zeros(n)
        # dict = self.alphas
        # for k,v in dict.items():
            # weights[k] = v
        # y_hat = np.sign(np.multiply(weights, kern))


# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples

    # Implement this!
    # TODO: after kernel perceptron is fixed, merge into budget perceptron
    def fit(self, X, Y):
        # update step
        if y_true * y_hat <= self.beta:
            SV_indices.append(t)
            alphas.update({t:y_true})
            # removal step
            if len(SV_indices) > self.N:
                foo = [ Y[i] * (y_hat[i] - alphas[i]*np.dot(X[i].T, x[i])) \
                        for i in SV_indices ]
                arg_maxmarg = foo.index(max(foo))
                SV_indices.pop(arg_maxmarg)


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
